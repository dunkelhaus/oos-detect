# CLINC/POG acceptance report

## Outcome and scope

The final isolated Colab Tesla T4 run passed all 258 reproduction tests, the
runtime CLI, all seven tiny CLI smoke cases, and a fresh-process generator CLI
resume. A later T4 run also passed the full-dimensional batch-32 FP32 check.
All 43 implementation tasks are complete. This is bounded, non-paper acceptance.
It does not verify a full paper campaign or published scores.

These are historical acceptance results for the source snapshots described
below. They do not verify the later checkpoint-policy, persistence, or campaign
runner changes. No new test run or full training is reported by this update.

The results below come from the approved Colab MCP session. No model or test
command was run on the local host to prepare this report. Cell 35
(`nnThQfooCJB-`) in the original acceptance notebook records these results. Its cells can be run
again. Their `/content` files and runtime are temporary, not persistent backups.

## Source and installation evidence

The isolated recursive checkout is `/content/oos-detect-isolated-3ii_xbht`.
Its environment is `/content/oos-detect-isolated-3ii_xbht-venv`.
Four verified private Git bundles supplied the root, POG, nested GAN, and dataset
repositories. The transfer included 61 source changes. It did not require
credentials in the bundles or a destructive replacement of existing checkouts.

| Repository | Base commit |
| --- | --- |
| Root | `71498f8be5d151487bad3a1dfcb8f839a6c94625` |
| POG | `e5aa1f2e3ad78d7580006a07f415a9a37dd628fc` |
| Nested GAN | `40355f9655a113d18e46e209854440cad8a18539` |
| Dataset | `321db5c8b98ac3e54fa9c38a2bd0d2170c1028ff` |

These commits alone do not identify the tested implementation: source changes
remain dirty and uncommitted. The isolated copy included those changes and the
approved nested `.gitmodules` repair. No commit or push was made. Original
datasets, the paper PDF, unrelated floor-plan material, and `labeled_data` ZIP
uploads were left unchanged.

The locked installation included all three editable distributions:
`oos-detect`, `pseudo-ood-generation`, and `wgan-gp`. The initial 44-package sync
took 61.204 seconds. The final isolated cached sync took 2.1433 seconds; this is
not a cold-install measurement.

## Executed commands and measurements

These normalized shell forms show the commands executed through notebook subprocesses.
`UV_BIN` is the uv executable from the dedicated tool environment described in
[runtime setup](runtime.md).
`snip` is the optional output wrapper used by the documentation, not a project
dependency. Each Python command used the isolated interpreter in a fresh
process, with the isolated checkout as its working directory.

```bash
export UV_PROJECT_ENVIRONMENT=/content/oos-detect-isolated-3ii_xbht-venv
export UV_PYTHON_INSTALL_DIR=/content/oos-detect-python
snip "$UV_BIN" sync --locked --python 3.12 --project /content/oos-detect-isolated-3ii_xbht
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/reproduction -q
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce runtime --device cuda:0
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce smoke --output /content/clinc-isolated-acceptance-v6uctvej/smoke --device cuda:0
```

| Runtime item | Observed value |
| --- | --- |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cu132 |
| CUDA runtime | 13.2 |
| GPU | Tesla T4, SM 7.5 |
| NVIDIA driver | 580.82.07 |
| Precision | FP32 |

| Check | Result | Measured time | Peak allocated CUDA memory |
| --- | --- | --- | --- |
| Reproduction suite | 258 passed | 44.56 s; 46.3809 s outer elapsed | Not separately reported |
| Runtime CLI | LSTM, convolution, second-order GP backward passed | 0.3461 s | 17,061,376 bytes |
| Seven-case CLI smoke | All seven passed | 7.041345411 s | 25,843,200 bytes (about 24.65 MiB) |
| Generator CLI resume | Boundary 4 → 20 passed | 9.778 s outer elapsed | Not separately reported |

Peak allocated memory is not total GPU memory use or reserved memory. These
tiny measurements do not estimate full-profile time, memory, or disk use.

The smoke covered MSP, Perturb, Mix, POG, AEPOG, `pog-no-noise`, and
`pog-no-soft-token`. It wrote artifacts under
`/content/clinc-isolated-acceptance-v6uctvej/smoke`. It exercised synthetic
preparation, generator updates where applicable, independent checkpoint-driven
generation, final CNN updates, evaluation, validation-only calibration, and
aggregation. The smoke used reduced non-paper settings, including batch size
two and random-only embeddings. The separate batch-32 result follows below.

The suite covers the four capability contracts: CLINC split/vocabulary/data
provenance, fork-owned latent WGAN-GP, token POG/AEPOG training, and final
classification/evaluation/run state. It also covers supported import boundaries,
all seven tiny API artifact flows, generated UNK identity, parameter isolation,
loader/RNG restoration, checkpoint compatibility, selection provenance, and
fresh-process CUDA initialization. Supported imports do not require legacy,
image, or online-tracking dependencies.

## Full-dimensional batch-32 T4 check

After the runtime changed, the source snapshot and locked environment were
restored in a fresh Colab T4 session. The snapshot contained 68 verified files.
Its SHA-256 was `6be981eac424758ef48857fac2337f39646e0482e60bb46947f53a008ce7e509`.
Pinned CLINC originals were downloaded again and checked against their recorded
hashes. This source-only restore did not repeat the recursive Git checkout test.
The earlier isolated checkout provides that separate evidence.

Python 3.12.14, PyTorch 2.14.1+cu132, and CUDA 13.2 were installed with
`uv sync --locked`. The native Colab environment was not replaced. The check used
FP32, batch size 32, 64-token inputs, 63-step rollouts, and 150 intent outputs.
It retained the full model dimensions: 100d embeddings, codes and noise;
512-wide GAN hidden layers; and four CNN widths with 128 filters each and
512-wide classifier hidden layers. Vocabulary sizes were 5,078 and 7,404.
No batch reduction was needed.

One complete generator cycle and one independent final CNN update passed for
each mode. Each ordinary cycle included five critic updates. AEPOG also ran
the unlabeled Dmix reconstruction and critic updates. Losses and gradients were
finite. The runtime diagnostic also passed LSTM, convolution, and second-order
gradient-penalty backward checks.

| Mode | Generator cycle | Final CNN update | Peak allocated CUDA memory |
| --- | --- | --- | --- |
| POG | 0.8683 s | 0.0387 s | 629,000,192 bytes |
| AEPOG | 0.2526 s | 0.0285 s | 629,000,192 bytes |
| POG without noise | 0.2123 s | 0.0260 s | 628,987,392 bytes |
| POG without soft tokens | 0.2233 s | 0.0253 s | 626,995,712 bytes |

The maximum was about 600 MiB. These cycle and update times exclude model
construction. Total internal time was 9.3402 seconds; outer process time was
17.4828 seconds. These are single-cycle checks, not full-training estimates.
Random-only embeddings make this an explicit non-paper check, despite its
full model dimensions. No pretrained-vector result or full campaign is claimed.

The current notebook cell `rviSnUjmOygy` records this run. Its JSON artifact is
`/content/clinc-batch32-bs3_l9te/result/acceptance.json`. The script SHA-256 at the time of that run was
`984556ba5814413b55e077e10512ab57e8031c8c6a159bc4adeebaff414193e1`.
This historical digest does not identify the current edited script.
To repeat the bounded check from a prepared recursive checkout, use a new output
directory:

```bash
snip uv run --locked python tests/reproduction/batch32_smoke.py --output runs/batch32-acceptance --device cuda:0
```

## Generator CLI resume evidence

The API created a partial tiny POG run and stopped at completed boundary 4.
A fresh interpreter then ran this command form. `PREPARED` denotes the same
prepared smoke artifact; `RUN` denotes that partial generator run. These are
run-specific paths from the notebook, not requests to create replacement inputs.

```bash
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce train-generator --prepared "$PREPARED" --output "$RUN" --profile tiny-smoke-non-paper --variant POG --mode pog --seed 0 --resume "$RUN/checkpoints/boundary-000000004.pt"
```

The CLI continued from boundary 4 to boundary 20. All five phase counts ended
at four updates each. The log contained exactly 20 entries, and the SHA-256
digests of older checkpoints were unchanged. Completed work was not repeated.
This was a real generator CLI resume, not only a parser or API check.

Classifier epoch resume passed the API tests. A separate fresh-process
classifier CLI resume was not run. Neither resume result promises bitwise
replay across GPU models or software releases.

## Remaining limits and assumptions

- The user approved Colab Tesla T4 as the acceptance target.
  The locked T4 runtime and batch-32 checks satisfy
  tasks 1.3 and 10.2. No CPU model fallback was used.
- No full paper-profile training or seven-variant, ten-seed campaign was run.
  There are no measured full-run scores, duration, or storage requirements.
  Private IPA/Bixby and Chat datasets remain outside the reproduced scope.
- Paper classifier vocabulary uses declared training, validation, and test
  **text**, without held-out labels. This text-only transductive assumption is
  not a train-only vocabulary protocol. Generator vocabulary uses IND training
  text only. See [data protocol](data-protocol.md).
- GloVe 6B 100d is a chosen source assumption, not a paper-prescribed source.
  Paper runs require the user-supplied file and its recorded hash and coverage.
  Random-only smoke embeddings do not verify pretrained paper training.
- Reports retain both IND-positive and OOD-positive AUPR and FPR orientations.
  The paper designates AUPR as OOD-positive, but its FPR orientation is ambiguous.
  AUPR also depends on class balance. Comparisons are provisional, not a
  published-score guarantee. See [metric definitions](metrics.md).
- Checkpoint-policy defaults still save full state after every generator phase
  and retain all files. The campaign caller explicitly selects a 60-second
  interval and bounded retention. Epoch completion, an explicit stop, and
  completion always save; classifiers always save completed epochs. Latest,
  selected, and final files stay available where applicable. Pruning is limited
  to owned files with matching ledger hashes. Resume reconciles logs with the
  saved boundary. See [checkpoint policy](checkpoints.md#operational-checkpoint-policy).
  Measure disk capacity and checkpoint growth; the tiny smoke does not establish
  campaign storage needs. Load only trusted checkpoints.
- The backup callback runs on the training thread with training paused. The
  campaign caller selects 600-second backups and a completion backup. Keep active
  files on the local Colab SSD and verified copies on Drive. See
  [persistent run backups](persistence.md). This is not evidence that backups
  or new resume behavior were tested in the historical acceptance run.

## Deliberate full-campaign launch

Do not execute full training as setup or acceptance. The user approved a
seed-major campaign of seven methods for seeds 0–9. The first launch is
**MSP seed 0 only**. Check disk and GPU capacity, supply GloVe, and prepare the
original full CLINC sources. Use the
[preferred campaign runner](reproduction-commands.md#preferred-campaign-runner-first-launch-only)
once it is ready. Its planned command selects one paper experiment, 60-second
checkpoints, and 600-second backups. It has not been executed for this report.
The old `set -e` loops are reference only, not a safe start or restart procedure.

The following older POG stage example is retained for reference. It is **not**
the approved MSP seed-0 first launch and was **not executed** during acceptance:

```bash
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce train-generator --prepared artifacts/clinc-paper --profile paper --variant POG --mode pog --device cuda:0 --generator-epochs 80 --classifier-epochs 30 --seeds 0 1 2 3 4 5 6 7 8 9 --seed 0 --embeddings /content/glove.6B.100d.txt --output artifacts/pog/seed-0/generator
```

The `--seeds` list declares the campaign; it does not launch the other seeds.
Do not use the linked legacy loops to restart existing runs. Preserve per-seed
reports and missing-seed status. Do not infer numerical parity from test success
or seed completeness. Saving the notebook to Colab/Drive is not publication.
Exclude the external OpenSpec planning folder from publication.

## Reference checks

The command and checkpoint guidance was checked against the official
[uv command reference](https://docs.astral.sh/uv/reference/cli/#uv-sync)
(`--locked` preserves the lock) and the
[PyTorch checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training)
(resume requires optimizer state as well as model state). These references
explain the interfaces; the Colab measurements above are the execution evidence.
