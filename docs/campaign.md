# One-run Colab campaign runner

Use `python -m oos_detect.reproduce.campaign` to run one paper experiment and
seed. The runner does not advance automatically. The order is MSP, Perturb,
Mix, POG, AEPOG, POG without noise, and POG without soft tokens at seed 0;
repeat that order for seeds 1 through 9. These are repeated seeds, not folds.

## Before launch

- Use a clean, pinned recursive checkout and the locked Python environment.
- Mount Google Drive. Check its account quota separately from local SSD space.
- Supply the verified GloVe 6B 100d file. Do not substitute random vectors.
- Use one active runtime per experiment and seed. A local file lock prevents
  duplicate writers on the same runtime, not on separate Colab runtimes.

For the first run:

```bash
snip uv run --locked python -m oos_detect.reproduce.campaign --experiment MSP --seed 0 --work-root /content/oos-detect-campaign-work --backup-root /content/drive/MyDrive/oos-detect-campaign/backups --embeddings /content/clinc-campaign-assets/glove.6B.100d.txt --backup-seconds 600 --checkpoint-seconds 60 --device cuda:0
```

This uses the paper profile: 30 final classifier epochs, or 80 generator epochs
and then 30 classifier epochs for generator methods. It preserves the full
data, architecture, optimizer, and seed settings. `--seed` selects one run.
Neither this command nor a completed run launches the other 69 classifiers.

## Storage and recovery

Training and checkpoint publication use the local SSD. Do not point `work-root`
at Drive: the checkpoint writer requires local hard links. Generator checkpoints
are saved at completed phases about every 60 seconds and at epoch boundaries.
Classifier checkpoints are saved at completed epochs. Retention keeps the latest
resume checkpoint and selected/final model references. It does not delete unknown
or modified checkpoint files. All phase logs remain available.

The synchronous training callback creates a gzip level-1 snapshot at a safe
checkpoint. It copies and verifies the snapshot on Drive before publishing a
completion manifest. It keeps two verified generations per run, including all
retained models, optimizer state, RNG and loader state, configs, logs, and results.
Backups occur at the first checkpoint, about every ten minutes at a safe
checkpoint, and at stage completion. See [backup details](persistence.md).

After reclamation, recreate the same pinned checkout and environment, restore
the embedding asset, and rerun the same command. The runner restores the newest
valid matching snapshot to local storage and resumes its saved boundary. It
rejects source, data, config, embedding, or recorded runtime mismatches. Modified
or unknown runs are preserved, not reset. Work after the last verified backup
can be lost; the window also includes any delay to the next safe checkpoint.

The reusable notebook is saved separately in Colab/Drive, not in this repository.
It contains clone, environment, asset, restore-check, launch, and progress cells.
Its default request is MSP seed 0. Keep notebook, planning, credentials, assets,
and run artifacts out of source commits.

## Progress and estimates

Read `campaign.jsonl` in the selected local run. Checkpoint events report file
size, completed epochs, measured wall time, and a remaining-time estimate.
The wall time includes validation and checkpoint I/O; subsequent measurements
also include intervening backups. Phase logs separately record checkpoint
serialization time. Backup events report the verified generation and logical
file bytes. The completion manifest identifies the actual compressed archive.

The initial 100–175 GPU-hour campaign estimate was not measured. Use actual
epoch and backup times before refining it. Tiny validation snapshots do not
measure full-model backup costs or the account's available Drive quota.

## Pre-publication validation

On the locked Python 3.12.14 / PyTorch 2.14.1+cu132 Colab T4 environment,
331 tests passed in 45.36 seconds. The notebook JSON schema and code cells
validated. A mounted-Drive check verified 42 restored files, then continued
a non-paper generator from boundary 4 to 40 and a classifier from epoch 1
to 2. The receipt is under Drive's campaign `validation` directory with ID
`ac7de2cfb7804bcd88a3cefc582112c8`. This was a bounded recovery check, not a
paper training run or a published-score result.
