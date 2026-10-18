# Experiment profiles

## Status

These profiles are integrated with the artifact-stage CLI. They do not start a
campaign. The locked Colab T4 runtime passed all 258 reproduction tests and the
seven-case CLI smoke. See [reproduction commands](reproduction-commands.md) for
CLI examples and ablations, and the [acceptance report](acceptance-report.md)
for evidence. Private paper datasets, full campaigns, and paper
scores remain unverified.

## Select a profile

Use `oos_detect.reproduce.config`. This module uses only the Python standard
library. It does not import PyTorch, create tensors, or check CUDA availability.
Model code must still check the selected CUDA device. There is no CPU fallback.

```python
from oos_detect.reproduce.config import (
    load_config, paper_config, smoke_config, validate_prerequisites,
)

campaign = load_config("configs/reproduction/paper-pog.json")
run_settings = campaign.for_seed(0)
generator_settings = campaign.pog_settings()  # POGSettings(**generator_settings)
model_settings = campaign.model_kwargs()     # build_model(..., **model_settings)
classifier_settings = campaign.classifier_settings(0)  # ClassifierConfig(**...)
```

`resolve_config(mapping)` rejects unknown keys and invalid values. `load_config`
also rejects duplicate JSON keys and non-finite JSON constants. `to_dict()` returns
all resolved settings. Save this result in the run manifest. `for_seed(seed)` adds
the selected seed for `state.create_run`; it does not start a run or derive RNG
streams. Use the existing `state.derived_seed` streams for preparation,
initialization, loaders, perturbation, generation, and evaluation.

| Selection | Classifier variant | Generator mode | Entropy source |
| --- | --- | --- | --- |
| MSP | `MSP` | unused `pog` | None |
| ER+Perturb | `Perturb` | unused `pog` | Perturbed IND training tokens |
| ER+Mix | `Mix` | unused `pog` | Unlabeled Dmix |
| ER+POG | `POG` | `pog` | Exported POG text |
| ER+AEPOG | `AEPOG` | `aepog` | Exported AEPOG text |
| POG without noise | `POG` | `pog-no-noise` | Separately exported ablation text |
| POG without soft tokens | `POG` | `pog-no-soft-token` | Separately exported ablation text |

Use `paper_config(variant)` or `smoke_config(variant)` for any baseline. For an
ablation, use `paper_config("POG", mode="pog-no-noise")` or
`paper_config("POG", mode="pog-no-soft-token")`. The smoke function accepts the
same modes. AEPOG mode and variant must agree. Ablations cannot consume Dmix.
The unused generator settings on MSP, Perturb, and Mix do not permit generator
training. Their generator stages fail validation.

## Paper settings

The paper profile fixes generator training at 80 epochs, classifier training at
30 epochs, seeds at 0–9, and corpus size at 15,000 texts. It uses 150 intent
outputs, 100d embeddings/codes/noise, four GAN affine layers with 512-wide hidden
layers, CNN widths 2/3/4/5 with 128 channels, and 512-wide classifier hidden
layers. The token limit is 64, including BOS/EOS. Rollouts predict at most 63
tokens after BOS.

The following are recorded engineering assumptions, not paper claims:

- Batch size 32; Adam rates 0.001 for AE, auxiliary CNN, and final CNN.
- Adam rates 0.0001 and betas (0, 0.9) for generator and critic.
- GP weight 10, five ordinary critic updates, entropy weight 1, AE clip norm 5.
- AEPOG adds one Dmix reconstruction update and one Dmix critic update per cycle.
- Linear training temperature decay from 1 to 0.1; standard Gaussian code noise.
- REINFORCE baseline decay 0.9 for the no-soft-token ablation.
- Generation temperature 1, EOS stopping, retained duplicate text, bounded empty
  retries (100), and unique record IDs.
- Final entropy weight 1 and Perturb replacement probability 0.5.
- GloVe 6B 100d supplied by the user; seeded initialization for missing tokens.

No reduced dimensions, counts, epochs, seed sets, random-only embeddings, or
training subset limits are accepted under the paper label. The no-noise mode
changes the trainer's reconstruction-noise switch, not the base noise setting.
The no-soft-token mode uses the trainer's sampled-token policy gradient, not a
straight-through argmax estimator.

The generator vocabulary uses only IND training text. The classifier vocabulary
uses declared training, validation, and test text without labels. This is a
transductive text-only assumption. Dmix contains 10,000 IND training records and
250 OOD-plus training records. Dmix supplies no supervised targets. Select the
final generator epoch. Select the final classifier by best IND validation
accuracy, with the earliest epoch on a tie. Do not select with test labels.

## Prerequisite checks

Call `validate_prerequisites(config, stage, ...)` before loading models. Stages are
`prepare`, `train-generator`, `generate`, `train-classifier`, and `evaluate`.
Text preparation does not require embeddings. Paper training requires an existing
embedding file passed as `embedding_path`. There is no automatic download or
random fallback. The existing `data.load_embeddings` must then check vector
dimensions, finite values, vocabulary coverage, and source hashes.

For generated-text classifier variants, pass `generated_manifest`,
`generated_records`, and expected `data_sha256`, `vocabulary_sha256`, and
`checkpoint_sha256`. Expected hashes must come from the selected artifacts, not
from the corpus being checked. Each hash must be a lowercase SHA-256 value.
The manifest must include those hashes plus `variant`, `mode`, `profile`,
`epoch`, `epochs`, and `count`. Both epoch fields must equal the configured final
generator epoch. The generated artifact profile must be `paper-generation` for a
`paper` campaign or `tiny-non-paper-generation` for a `tiny-smoke-non-paper`
campaign, as written by `generation.export_generated_corpus`. Campaign labels
and prepared-data profile labels are not valid generated artifact profiles.
Records must contain only `id` and nonempty `text`. IDs must be
unique; duplicate text is allowed. Both the manifest count and actual record
count must match. The artifact reader must check file hashes before this call.
The pipeline first checks the original hashed export, including its mechanism
variant and generation-stage profile. For prerequisite validation, it changes
only the in-memory variant to the classifier family (`POG` for either POG
ablation); it retains the generation-stage profile. The classifier receives a
separate consumer view with the classifier variant and campaign profile. That
view stores the original hash as `export_manifest_sha256`; the original manifest
is retained separately in run inputs. Do not pass the consumer view to
`validate_prerequisites`. The final classifier also checks its entropy source.

Mode and profile prevent reuse of standard POG text in an ablation or reuse of
smoke text in a paper run. Corpus readers must preserve the original manifest.
This validator does not read checkpoints, prove that
training finished, or replace prepared-data/Dmix validation.

## Tiny smoke settings

`configs/reproduction/tiny-smoke-pog.json` is explicitly non-paper. It uses seed
0, one epoch per model, batch size 2, two intent classes, 8d codes/noise, small
hidden layers, eight generated texts, and eight-token inputs. Training and
validation limits are 8 and 4. Dmix fixture counts are 4 IND and 2 OOD records.
Embeddings remain 100d because the current embedding loader supports 100d only.
Random-only initialization is explicit and cannot be combined with a pretrained
path. Use a two-intent fixture; do not truncate full CLINC and retain invalid
150-class targets. The integrated pipeline enforces the declared subset limits
and validates the fixture split contract. Custom callers must do the same.
These profiles do not change the original data.

Colab Tesla T4 is the user-selected CUDA acceptance target. The tiny smoke
supports bounded acceptance on that GPU, not paper-score parity or a
paper-dimension batch-32 FP32 result.
Run the ten-seed campaign only through a separate, explicit request.

## API references

- [Python 3.12 JSON](https://docs.python.org/3.12/library/json.html)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
