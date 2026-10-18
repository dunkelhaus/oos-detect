# CUDA runtime

## Install

Run setup and verification through the approved Colab MCP session, not the local
host. Use Linux x86-64 and an NVIDIA driver. The project uses Python 3.12 and the
official PyTorch 2.14.1 CUDA 13.2 wheel. The wheel provides the CUDA runtime.
The supported operations do not need a local CUDA compiler.

Use a recursive checkout at `/content/oos-detect-transfer`. Install `uv` with pip
in a dedicated project tool environment, as described in the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/),
not with an OS package manager. Keep the locked project environment separate
from the Colab kernel environment with
[`UV_PROJECT_ENVIRONMENT`](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path):

```bash
snip git -C /content/oos-detect-transfer submodule update --init --recursive
snip python3 -m venv /content/oos-detect-toolenv
snip /content/oos-detect-toolenv/bin/python -m pip install uv==0.12.21
export UV_BIN=/content/oos-detect-toolenv/bin/uv
export UV_PROJECT_ENVIRONMENT=/content/oos-detect-locked-venv
export UV_PYTHON_INSTALL_DIR=/content/oos-detect-python
snip "$UV_BIN" python install 3.12
snip "$UV_BIN" sync --locked --python 3.12 --project /content/oos-detect-transfer
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce --help
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m oos_detect.reproduce runtime --device cuda:0
snip "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/reproduction
```

Run these shell examples with `/content/oos-detect-transfer` as the working
directory. If `snip` is unavailable in Colab, omit only that prefix. Each Python
command starts a fresh process. Do not import model packages into the notebook
kernel to verify the isolated environment. For separate notebook subprocess
calls, pass `UV_PROJECT_ENVIRONMENT` explicitly and use its Python executable;
shell exports do not persist across independent calls.

## Verified locked T4 results

With uv 0.12.21, `uv python install 3.12` supplied Python 3.12.14. The isolated
locked sync resolved and installed 44 packages, including all three editable
packages and PyTorch 2.14.1+cu132. The initial sync took 61.204 seconds. The final
isolated recursive checkout at `/content/oos-detect-isolated-3ii_xbht` used
`/content/oos-detect-isolated-3ii_xbht-venv` and
`UV_PYTHON_INSTALL_DIR=/content/oos-detect-python`. Its cached locked sync took
2.1433 seconds. This checkout included the uncommitted implementation changes.

The FP32 runtime CLI passed LSTM, convolution, and second-order GP backward on
T4 SM 7.5, CUDA 13.2, driver 580.82.07: 0.3461 seconds and 17,061,376 bytes peak
allocated CUDA memory. All 258 reproduction tests passed in 44.56 seconds
(46.3809 seconds outer elapsed), including fresh-process CUDA initialization
regressions. The seven-case CLI smoke passed in 7.041345411 seconds with 25,843,200
bytes peak allocated CUDA memory.

Generator CLI resume also passed from boundary 4 to boundary 20. Classifier
epoch resume passed API tests; a separate classifier CLI resume was not run.
See the [acceptance report](acceptance-report.md) for commands and evidence.
Colab Tesla T4 is the user-selected CUDA acceptance target. These results verify
the locked runtime on T4 only. Private paper datasets, full campaigns, and
paper scores remain unverified.

## Device policy

All training, generation, inference, embedding tensor initialization, and model
tests require CUDA. Select the device with `--device cuda:0`. CPU devices are
not supported. The program fails if CUDA is not available; it does not use a
CPU fallback. File operations, tokenization, and report writing can use the CPU.

The initial precision is FP32. The runtime command checks LSTM backward,
convolution backward, and second-order gradient-penalty backward. It reports
the selected GPU, compute capability, runtime versions, elapsed time, and peak
allocated GPU memory. It does not change other GPU workloads.

## Package ownership

The root lock installs three editable packages:

- `oos-detect`: data preparation and experiment commands.
- `pseudo-ood-generation`: sentence models and POG training.
- `wgan-gp`: the existing nested fork, adapted for latent vectors.

The supported path does not import AllenNLP, Transformers, Prefect, pandas,
torchvision, imageio, or online tracking. The old BERT script is preserved in
`oos_detect/train/legacy_bert.py`. It is unsupported in this environment.
`python -m oos_detect.train` now exits with migration guidance and does not train.

The lock and source declarations follow the official
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
The [official wheel index](https://download.pytorch.org/whl/cu132/torch/)
contains the pinned Python 3.12 Linux wheel.
