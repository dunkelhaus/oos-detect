# POG training API

The supported package uses CUDA only. Import `build_model` and `build_trainer`
from `pseudo_ood_generation.builders`. Import `POGSettings` from
`pseudo_ood_generation.trainer`. These imports do not load AllenNLP or image tools.
The installed `wgan_gp` fork owns the generator, critic, GP, and GAN updates.

The integrated CLI handles prepared artifacts, phase checkpoints, and independent
generation. See [reproduction commands](reproduction-commands.md) for real stage
commands and both ablations. The trainer API below does not manage those files.

## Build and run a cycle

Use prepared generator and auxiliary vocabularies. Supply one auxiliary token ID
for each generator token in `token_mapping`. PAD must map to PAD. Supply pretrained
CUDA embedding tensors for a paper-profile run. Omitted embeddings mean random
initialization; this is suitable only for a non-paper smoke test.

```python
from pseudo_ood_generation.builders import build_model, build_trainer
from pseudo_ood_generation.trainer import POGSettings

model = build_model(generator_vocab_size, auxiliary_vocab_size, token_mapping,
                    generator_embeddings=generator_embeddings,
                    auxiliary_embeddings=auxiliary_embeddings, device="cuda:0")
trainer = build_trainer(model, POGSettings(mode="pog"), log_callback=write_jsonl)
trainer.set_epoch(0)
records = trainer.train_cycle(ind_tokens, ind_intents, rng=cuda_generator)
```

Inputs are CUDA tensors. `ind_tokens` is a right-padded long tensor `[B,T]`
with BOS and EOS. `ind_intents` is a long tensor `[B]` with IND class IDs only.
`cuda_generator` is an optional `torch.Generator` on the model device.
No batch is moved to CPU. No loader or file is opened by the trainer.

## Phase mapping

The Algorithm 1 POG cycle runs these separate optimizer steps in order:

| Method | Objective | Parameter owner |
| --- | --- | --- |
| `reconstruction_step(tokens)` | Masked next-token CE with noisy codes | AE |
| `auxiliary_step(tokens, labels)` | Intent CE on reconstructed tokens | Auxiliary CNN and table |
| `critic_step(tokens)` | Fake score minus real score plus weighted GP | Critic |
| `generator_wasserstein_step(batch_size)` | Negative fake score | Generator |
| `generator_entropy_step(batch_size)` | Negative predictive entropy | Generator |

The critic phase repeats `critic_iterations` times. Each repeat uses the same
real batch and fresh random noise. The two generator phases use the same Adam
optimizer and make successive steps. They are not a combined loss.

Auxiliary CE uses a new teacher-forced reconstruction after the reconstruction
update. Its soft tokens retain the AE graph. Only the auxiliary optimizer steps;
AE gradients are discarded. Real critic codes are clean and detached. Decoder,
auxiliary, and critic weights are frozen during generator entropy training, but
the input graph remains active. Soft decoding feeds expected AE embeddings back
at every step. The classifier receives expected auxiliary embeddings through the
saved vocabulary mapping. All phase gradients are cleared on exit, including
failure paths. Non-finite parameters, losses, gradients, and diagnostics fail.
Failures do not roll back an optimizer step that has already completed.

For Algorithm 2, select `mode="aepog"` and pass only an unlabeled token tensor:

```python
records = trainer.train_cycle(ind_tokens, ind_intents,
                              dmix_tokens=unlabeled_tokens, rng=cuda_generator)
```

AEPOG appends extra reconstruction and critic phases. Their repeat counts are
`dmix_reconstruction_iterations` and `dmix_critic_iterations`. Direct phase calls
use `dmix=True`. There is no Dmix label argument. All other modes reject Dmix
before a cycle updates any weights. The caller must supply the declared Dmix
records and cycle loaders as needed. Each cycle consumes one supplied batch from
each applicable source. Logs count all repeats and actual examples, not unique
examples. The trainer does not verify dataset provenance.

## Settings and assumptions

These are engineering defaults where the paper does not specify a value.

| Setting | Default |
| --- | --- |
| Batch size / generator epochs | 32 / 80 |
| AE / auxiliary Adam learning rate | 0.001 / 0.001 |
| Generator / critic Adam learning rate | 0.0001 / 0.0001 |
| Generator / critic Adam betas | (0, 0.9) |
| GP weight / critic repeats | 10 / 5 |
| Entropy weight / AE clipping norm | 1 / 5 |
| Extra AEPOG reconstruction / critic repeats | 1 / 1 |
| Temperature start / floor | 1 / 0.1 |
| Rollout steps after BOS | 63 |
| REINFORCE baseline decay | 0.9 |

The AE and auxiliary optimizers use the PyTorch Adam default betas and epsilon.
There is no critic weight clipping. Batch size and epoch count describe caller
configuration; the trainer does not split batches or start epochs itself.
Call `set_epoch` with a zero-based epoch. Temperature decays linearly and reaches
the floor at the last configured epoch. A one-epoch smoke run uses temperature 1.
Builder defaults use 100d embeddings/codes/noise, 512-wide four-layer GAN MLPs,
and a 150-class CNN. The AE uses standard Gaussian reconstruction noise.
The builder exposes smaller dimensions and a noise standard deviation for tests.

Soft rollouts use a fixed length with all positions active; there is no hard EOS
stop. PAD and BOS cannot be generated. Sampled rollouts include the EOS decision
in the mask and exclude later positions. The 63-step rollout plus initial BOS
gives the default 64-token cap. Teacher-forced masks exclude PAD targets.

## Ablations

- `pog-no-noise` disables only reconstruction-code noise, including auxiliary
  reconstructions. All soft-token and GAN phases remain active.
- `pog-no-soft-token` samples reconstructed discrete tokens for auxiliary CE and
  detaches them. Generator training samples autoregressive discrete tokens. The
  loss is negative detached entropy advantage times summed masked sequence log
  probability. Frozen decoder log probabilities retain generator-code gradients.
  There is no straight-through estimator. The reward baseline starts at zero.
  Each successful policy update then sets `b = decay*b + (1-decay)*mean(reward)`.

Both ablations reject Dmix. `settings.reconstruction_noise` and
`settings.soft_tokens` identify the active mechanisms.

## Diagnostics and checkpoint integration

Each phase returns a record and calls the optional `log_callback(record)`.
Records contain phase, epoch, temperature, actual batch size, cumulative phase
updates and examples, loss, and pre-clipping gradient norm. Entropy phases also
report predictive entropy and the baseline used for that update. Critic records
include GP and mean real/fake scores. `train_cycle` returns records in phase order.

Save `model.state_dict()` and `trainer.state_dict()` separately. Trainer state
contains all four optimizer states, settings, epoch, counts, and policy baseline.
`load_state_dict` rejects different settings. The parent must also save and restore
RNG states, loader states, data hashes, and its completed-phase position. Phase
methods permit boundary resume without repeating a full cycle. The trainer does
not save files or validate external checkpoint manifests.

The parent pipeline owns checkpoint timing and retention. Its defaults save
every generator phase and keep all files. The campaign caller explicitly sets
`checkpoint_interval_seconds=60` and `retain_checkpoints=True`. It saves at a
completed phase after 60 seconds, and always at epoch completion, an explicit
stop, or completion. Unsaved phases can be repeated after interruption.

Bounded retention preserves latest, selected, and final references where
applicable. It prunes only owned files with matching ledger hashes. Resume
reconciles phase logs with the saved boundary. The backup callback runs on the
training thread with training paused, not in a background thread. See
[checkpoint policy](checkpoints.md#operational-checkpoint-policy) and
[persistent run backups](persistence.md).

## CUDA acceptance

Run through the approved Colab runtime from the repository root:

```bash
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/reproduction/test_pog_training.py -q
```

The tests require CUDA and never skip model checks. They cover each phase owner,
AE-gradient discard, clean detached critic codes, finite nonzero entropy and
policy gradients, every mode, counts, Dmix rejection, baseline continuation,
temperature, and non-finite guards. Use the isolated environment from
[runtime setup](runtime.md). The earlier locked Colab T4 runtime passed all 258
reproduction tests and the seven-case CLI smoke. See the
[acceptance report](acceptance-report.md). Generator CLI resume passed from POG
boundary 4 to boundary 20 in a fresh process. Classifier epoch resume passed API
tests, but a separate classifier CLI resume was not run. Private
paper datasets, full campaigns, and paper scores remain unverified. These
historical results do not verify the later checkpoint-policy or campaign-runner
changes. No new tests or full training are reported here. Check disk capacity
before training; periodic checkpoints and bounded retention do not establish
full-campaign storage needs. The approved first launch is MSP seed 0 only.
