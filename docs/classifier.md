# Final intent classifier API

`oos_detect.reproduce.classifier` provides a reusable API. It does not start a
command, read a run directory, write artifacts, or download resources. All model
operations, token tensors, optimizer tensors, and prediction tensors use CUDA.
CPU execution has no fallback.

The integrated CLI reads artifacts and writes run outputs through
`oos_detect.reproduce.pipeline`. See [reproduction commands](reproduction-commands.md)
for stage commands and ablations.

## Configuration and initialization

Use `ClassifierConfig` to select `MSP`, `Perturb`, `Mix`, `POG`, or `AEPOG`.
Use `FinalClassifierTrainer(config, vocabulary, embeddings=None)` to create a
fresh final classifier and Adam optimizer. Alternatively, use
`build_final_classifier(config, vocabulary, embeddings=None)` for the model only.

The model is the existing `SentenceCNN` architecture, not a copy of an auxiliary
model. It has its own word table and CNN parameters. The builder accepts an
`EmbeddingResult` from `data.load_embeddings`, not an auxiliary model or its
state. It copies the initial word vectors. The supplied vectors do not change.
Initialization uses the derived initialization seed and preserves the caller's
RNG state.

Paper defaults are 100-dimensional embeddings, convolution widths 2/3/4/5 with
128 channels each, two 512-unit hidden affine layers, and 150 intent outputs.
There is no OOS output. Paper mode requires a matching pretrained embedding
manifest with a source hash. Missing embeddings cause an error, not random
initialization or a download. GloVe is a documented source assumption, not a
paper-prescribed source.

The configuration defaults to 80 generator epochs, 30 classifier epochs,
`alpha=1`, perturbation probability 0.5, batch size 32, and Adam learning rate
0.001. These optimizer and sampling settings are recorded engineering
assumptions. Paper validation fixes the epoch counts, dimensions, alpha, and
perturbation probability. Smaller dimensions or shorter schedules require
`paper=False`. Random embeddings require non-paper mode. The parent experiment
runner must select ten explicit seeds, normally 0–9; this API trains one seed.

## Objective and sources

`classifier_objective(ind_logits, targets, entropy_logits=None, alpha=1.0)`
returns `loss`, `ce`, and `negative_entropy` tensors:

```math
L = CE(z_{IND}, y_{IND}) + \alpha\,\operatorname{mean}\left[\sum_k p_k\log p_k\right].
```

`negative_entropy(logits)` returns the second term without alpha. Minimizing
this negative quantity favors a uniform intent distribution. Entropy inputs
never supply cross-entropy targets.

Call `make_entropy_source(config, vocabulary, train=..., dmix=None,
generated=None, data_sha256=None, checkpoint_sha256=None)` before training.
It rejects missing sources, extra sources, exposed targets, and mismatched
source policies.

| Variant | Supervised source | Entropy source |
| --- | --- | --- |
| MSP | `full:train` IND | None; factory returns `None` |
| Perturb | `full:train` IND | A separate stream of perturbed IND training text |
| Mix | `full:train` IND | `Mixture.records` from `data.build_dmix` |
| POG | `full:train` IND | Exported POG text, not latent codes |
| AEPOG | `full:train` IND | Exported AEPOG text, not latent codes |

Perturb replaces each non-special token with probability 0.5. Replacement IDs
are uniform over the non-special classifier vocabulary. PAD, UNK, BOS, and EOS
do not change. A replacement may equal its original token; the observed changed
fraction is therefore less than 0.5. `perturb_tokens(tokens, vocab_size,
generator=..., probability=0.5)` is also available separately.

Mix accepts only text-only `UnlabeledRecord` rows. It checks the ordered and
unordered membership hashes, counts, and training partition IDs. Each IND row
must match supervised training text by source ID. Paper Mix requires 10,000 IND
and 250 OOD training rows. IND overlap with supervised training is intentional.
The original OOD labels/status are not exposed to the objective.

Generated text uses `GeneratedCorpus(records, manifest)`. Each record must be
an `UnlabeledRecord` with a unique ID. Duplicate text is allowed. The manifest
must contain these fields:

| Field | Required value |
| --- | --- |
| `variant` | `POG` or `AEPOG`, matching the classifier variant |
| `epoch`, `epochs` | Both equal `config.generator_epochs` |
| `checkpoint_sha256` | Expected generator checkpoint hash supplied to the factory |
| `data_sha256` | Expected prepared-data hash supplied to the factory |
| `vocabulary_sha256` | Classifier vocabulary hash |
| `count` | Number of supplied text records; 15,000 in paper mode |

The integrated artifact reader verifies the actual file hashes and converts the
export manifest to this contract. It retains the original manifest separately.
Prerequisite validation uses its generation-stage profile; only the classifier
consumer view uses the campaign profile. Ablation variants become `POG` in that
view, with the original hash named `export_manifest_sha256`. See
[profile checks](experiment-profiles.md#prerequisite-checks). The trainer does not
read or hash checkpoint files.
`select_final_generator(checkpoints, configured_epochs=80)` returns the unique
metadata entry whose `epoch` equals the configured final epoch. It does not
inspect any accuracy or OOD metric.

## Training and loader exposure

Call `trainer.fit(train, validation, source=None, on_epoch=None)`. Train and
validation inputs must be `data.Record` sequences from `full:train` and
`full:val`, respectively. Test and OOD records are rejected. The method has no
test-data argument. Validation is read only for intent accuracy.

Each epoch runs the larger of the IND and entropy batch counts. Each shorter
loader restarts and reshuffles independently. Partial batches are retained.
Epoch boundaries reset loader iteration. Separate CUDA generators control IND
order, entropy order, and perturbation. Entropy order uses the loader stream
derived from `seed + 1`; the IND loader uses the configured seed.

History records include mean CE, mean negative entropy, IND-validation accuracy,
update count, actual IND/entropy example exposures, and both loader restart
counts. Means are weighted by the actual number of examples for each term.

Selection uses the best IND-validation accuracy. A strict improvement replaces
the selection; equal accuracy retains the earliest epoch. Training continues
through the configured final epoch. `ClassifierResult` contains
`selected_epoch`, `final_epoch`, `best_validation_accuracy`, `selected_state`,
`final_state`, and `history`. Both weight snapshots stay on CUDA and remain
separate from the live model. Load `result.selected_state` before test inference.
Do not use test labels to choose hyperparameters or generator artifacts.

## Checkpoint integration

The optional callback receives `(EpochEvent, trainer)` after each completed
epoch. An event contains `epoch`, `selected`, `final`, and `metrics`. The
trainer exposes `model`, `optimizer`, and `generators` for the existing
`reproduce.state` checkpoint API. It does not implement another serializer.

For example, the parent can save a new file for each selected or final epoch:

```python
from pathlib import Path
from oos_detect.reproduce.state import Boundary, save_checkpoint

def checkpoint_callback(event, trainer):
    if not (event.selected or event.final):
        return
    save_checkpoint(
        Path(run_directory) / f"classifier-epoch-{event.epoch}.pt",
        models={"classifier": trainer.model},
        optimizers={"classifier": trainer.optimizer},
        generators=trainer.generators,
        boundary=Boundary(event.epoch, 0, "classifier_epoch"),
        compatibility_hash=compatibility_hash,
        temperature=1.0,
        policy_baseline=None,
        extra={"classifier_selection": trainer.selection_state()},
    )
```

Keep the selected epoch file and final epoch file. They may be the same file
when the final epoch is selected. For interruption recovery after any epoch,
save every epoch instead of filtering the callback. Files must have new paths;
the state API rejects overwrites.

For resume, construct a trainer with the same resolved configuration. Call
`state.restore_checkpoint` with its model, optimizer, generators, and the parent
run's compatibility hash. Then call
`trainer.restore_selection(restored["extra"]["classifier_selection"])` before
`fit`. Selection state includes the retained selected weights. The next call
starts after the completed epoch, without repeating its work. Its history
contains only the newly completed epochs. The parent must bind the compatibility
hash to the configuration, data, source artifacts, and original embeddings.

## Prediction and verification

`predict_confidence(model, tokens)` returns CUDA tensors named `logits`,
`prediction`, and `confidence`. Confidence is the maximum softmax probability,
not an OOD score or a binary decision. Prediction does not build a gradient
graph and restores the model's previous training mode. Evaluation may use
`1 - confidence` as the OOD score.

`tests/reproduction/test_classifier.py` covers independent initialization, paper
validation, entropy sign, confidence, seeded perturbation, all five sources,
both cycling directions, source failures, forbidden test/OOD supervision,
final-generator selection, earliest ties, retained snapshots, and epoch resume
through the shared state API. The locked Colab T4 runtime passed all 258
reproduction tests and the seven-case CLI smoke. See the
[acceptance report](acceptance-report.md). Generator CLI resume passed from POG
boundary 4 to boundary 20 in a fresh process. Classifier epoch resume passed API
tests; a separate classifier CLI resume was not run. Private paper
datasets, full campaigns, and paper scores remain unverified.

The CLI supplies artifact reading, paper/smoke profiles, ablation names, and
persisted selection references. Each training command runs one explicit seed;
multi-seed campaigns require a separate execution request.
