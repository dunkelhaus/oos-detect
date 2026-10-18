# Independent pseudo-OOD generation

## Status

The Python API is available in `oos_detect/reproduce/generation.py`. The CLI
dispatches generation and both ablation training modes through the artifact
pipeline. See [reproduction commands](reproduction-commands.md) for commands and
verification limits. The locked Colab T4 runtime passed all 258 reproduction
tests and the seven-case CLI smoke. See the [acceptance report](acceptance-report.md).
Private paper datasets, full campaigns, and paper scores remain
unverified. Generator CLI resume passed from POG boundary 4 to boundary 20 in a
fresh process. Classifier epoch resume passed API tests; a separate classifier
CLI resume was not run.

## Checkpoint prerequisites

Use a trusted local checkpoint from `state.save_checkpoint`. Loading uses
`torch.load(weights_only=False)` because the existing format contains Python
and NumPy RNG state. Never load an untrusted checkpoint.

Build the model with the training architecture. Supply the original run
manifest, prepared manifest, and both ordered vocabularies. No training,
download, or optimizer step occurs during generation. Model weights are loaded
strictly. Training RNG and optimizer state are not restored.

At each completed generator epoch, put this metadata in the existing `extra`
field. Do not mark an incomplete epoch as complete.

```python
from oos_detect.reproduce.generation import checkpoint_generation_metadata

metadata = checkpoint_generation_metadata(
    mode="pog",  # also aepog, pog-no-noise, pog-no-soft-token
    completed_epochs=80,
    configured_epochs=80,
    data_sha256=prepared_manifest["manifest_sha256"],
    generator_vocabulary=generator_vocabulary,
    classifier_vocabulary=classifier_vocabulary,
)
# Add extra={"generation": metadata} to the existing save_checkpoint call.
# Preserve any other trainer metadata in extra.
```

Epoch counts in this metadata are completed counts, not zero-based trainer
indices. Only the final configured epoch is accepted. No validation or OOD
score selects a generator. The checkpoint container remains format version 1.
The integrated training writer supplies this metadata at completed epochs.
Custom API callers must also supply it. Older checkpoints without it fail rather
than assume that they contain the final epoch.

The default checkpoint model key is `pog`. Set `model_key="model"` if that
key was used at save time. Set `model_key=None` for checkpoints whose `models`
section was saved from `model.phase_modules`. The key sets must match exactly.

## Export API

Run the following in the approved CUDA environment, after the parent has
created the model and loaded the manifests and vocabularies:

```python
from oos_detect.reproduce.generation import GenerationConfig, export_generated_corpus

corpus = export_generated_corpus(
    "runs/seed-0/generated-seed-101",
    checkpoint="runs/seed-0/generator-final.pt",
    model=model,
    run_manifest=run_manifest,
    prepared_manifest=prepared_manifest,
    generator_vocabulary=generator_vocabulary,
    classifier_vocabulary=classifier_vocabulary,
    configured_epochs=80,
    config=GenerationConfig(seed=101),
)
```

Paper export requests exactly 15,000 records, temperature 1, and a 64-token
limit. It requires paper prepared data and the final configured 80th epoch.
This profile identifies export settings. It does not certify that an arbitrary
checkpoint used the full paper training architecture or pretrained embeddings;
the parent training configuration must enforce those requirements.

For a tiny independent check, use a separately saved tiny model checkpoint,
its matching prepared data, and explicit non-paper settings:

```python
config = GenerationConfig(seed=101, paper=False, count=12, batch_size=3, max_length=8)
# Pass config=config and configured_epochs=2 to export_generated_corpus.
# The prepared tokenizer limit must also be 8. Use a fresh output directory.
```

Artifacts are labeled `tiny-non-paper-generation` when `paper=False`, even if
some settings match paper settings. These snippets are API examples, not records
of executed commands.

## Sampling rules

- Use a private CUDA RNG from `derived_seed(seed, "generation")` for Gaussian
  latent noise and multinomial token sampling. Do not reuse training RNG state.
- Use evaluation mode and inference mode. Restore each module's prior mode.
- Exclude PAD and BOS from sampled outputs through the existing decoder API.
  Keep EOS eligible. Text stops at the first EOS. The current decoder still
  computes its fixed-length rollout after EOS; post-EOS tokens are not exported.
- Count the initial BOS in the length limit. Generate at most 63 output tokens
  for the 64-token profile. Do not force EOS at the limit. An output without EOS
  is counted as truncated.
- Join token strings with spaces. Retain literal `<unk>` and count UNK tokens.
  This is a recorded detokenization assumption, not natural punctuation repair.
- Retain duplicate text. Count each repeated accepted text after its first
  occurrence. Each record receives a unique ID.
- An immediate EOS is empty. Retry only empty slots, with fresh latent noise.
  Allow one initial attempt plus `max_empty_retries` attempts per slot. The
  API default retry count is 10; experiment profiles pass 100. Fail if any slot
  stays empty; never reduce the requested count or substitute another sampling
  policy.

Diagnostics count all attempts and empty outputs. Duplicate, UNK, and truncation
counts describe accepted records. Batch size affects random-number consumption
and is part of the replay settings. Replay requires the same checkpoint,
settings, software, and CUDA environment. Cross-device bitwise replay is not
promised.

## Artifact checks

The output directory must not exist. A successful export writes:

```text
generated-seed-101/
  corpus.jsonl
  manifest.json
```

Each corpus row contains only `id` and `text`. There is no intent target, OOD
class, or latent vector. Use these records for entropy regularization only.
The return value is the existing `classifier.GeneratedCorpus` type.

The manifest records checkpoint bytes, run compatibility, prepared-data and
both vocabulary hashes; mode and final epoch; seeds and settings; diagnostics;
runtime; and corpus bytes. `vocabulary_sha256` is the classifier vocabulary
hash for compatibility with `make_entropy_source`.
`generator_vocabulary_sha256` separately identifies the decoder vocabulary.
Do not confuse the data module's `content_hash` with the state module's
`fingerprint`: these functions use different JSON encodings. Preparation uses
the former; run and generation manifests use the latter.

`load_generated_corpus(directory, *, checkpoint_sha256, data_sha256,
vocabulary_sha256, generator_vocabulary_sha256, variant,
configured_epochs=80, paper=True)` checks expected identities, manifest and
corpus hashes, counts, unique IDs, and the exact target-free row schema.
Pass expected hashes from the intended run, not from an unverified export.
Missing artifacts fail with a prerequisite error. A failed export can leave a
reserved directory, but it has no success manifest and cannot be reused.

## Ablations

Training selects `pog-no-noise` or `pog-no-soft-token` through `POGSettings`.
The first disables reconstruction noise only. The second uses sampled-token
REINFORCE with the checkpointed reward baseline. Both exclude Dmix.
Generation uses the same sampled export policy for all four generator modes.
Mode and variant remain in the artifact metadata; generation cannot convert
one trained mechanism into another.

The CLI accepts both modes for separate training and generation stages. See
[ablation commands](reproduction-commands.md#deliberate-full-ten-seed-campaign).
Passing API integration tests does not verify those full campaign commands.

## API references

- [PyTorch checkpoint loading and trust rules](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)
- [PyTorch device-specific generators and seeds](https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html)
