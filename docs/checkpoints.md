# Local state and resume

Each run has an explicit output directory. A new run fails if that directory
already exists. The manifest records resolved configuration, data and embedding
hashes, independent seeds, Python/PyTorch/CUDA versions, the selected GPU, and
revision and dirty state for all three source repositories.

Checkpoints identify completed phase or epoch boundaries. The saved position
names the next phase. Arbitrary recovery in the middle of an optimizer step is
not supported. Save all participating models and optimizers, loader and other
RNG streams, global Python/NumPy/PyTorch/CUDA random state, temperature, and
the policy reward baseline. State is stored in local files only.

The integrated CLI writes generator phase checkpoints and classifier epoch
checkpoints. See [checkpoint and resume commands](reproduction-commands.md#checkpoints-and-resume)
for paths, artifact prerequisites, and explicit resume examples.

The state API is `oos_detect.reproduce.state`. Use `Boundary` to identify a
completed boundary. `save_checkpoint` rejects an existing destination.
`restore_checkpoint` checks the run compatibility hash before loading model
or optimizer state. That hash includes architecture, configuration, data, and
embedding identity. A different experiment must use a new directory.

Only load checkpoints that you trust. Full training-state restoration uses
Python pickle and is not safe for files from an untrusted source.

Same-environment continuation tests compare the next update after restoration
with an uninterrupted update on CUDA. This does not promise identical results
across different GPU models or software releases. Deterministic algorithms are
an explicit mode, not an implied property of a saved seed.

The earlier locked Colab T4 runtime passed all 258 reproduction tests, including API
phase/epoch continuation, and the seven-case CLI smoke. A fresh generator CLI
process also resumed POG from boundary 4 to boundary 20 without repeating work
or changing older checkpoint hashes. Classifier epoch resume passed API tests;
a separate classifier CLI resume was not run. See the [acceptance report](acceptance-report.md).
Private paper datasets, full campaigns, and paper scores remain
unverified.

These are historical results. They do not verify the later checkpoint-policy
or campaign-runner changes. No new test run or full training is reported here.

## Operational checkpoint policy

The pipeline functions `train_generator` and `train_classifier` accept these
options. They do not change experiment compatibility or model settings.

| Option | Default | Campaign caller |
| --- | --- | --- |
| `checkpoint_interval_seconds` | `None` | `60` |
| `retain_checkpoints` | `False` | `True` |
| `on_checkpoint` | `None` | Synchronous backup callback |

The defaults preserve the old behavior: save every generator phase and retain
all checkpoint files. A positive interval saves at the first completed phase
after the interval has elapsed. It is not a background timer or a strict
60-second recovery guarantee. Generator epoch completion, an explicit stop,
and training completion always force a save. Classifiers always save each
completed epoch; the interval cannot skip epoch or selection saves.

Despite its name, `retain_checkpoints=True` enables bounded retention and
pruning. `checkpoint-policy.json` records owned filenames, SHA-256 digests, and
latest, selected, and final references, where applicable. Referenced files stay
available. Pruning removes only unreferenced owned `boundary-NNNNNNNNN.pt`
files whose hashes still match. It leaves symlinks, changed files, and unrelated
files alone. A new checkpoint needs free space before old files can be pruned.
The disk guard reserves a full new checkpoint plus headroom; it does not prove
that a full campaign will fit.

Resume reconciles `phases.jsonl` with the saved boundary. It removes this stage's
later or duplicate records and repairs a missing last record from the checkpoint.
It preserves unrelated lines and rejects malformed JSON. Unsaved updates can be
repeated after interruption. Arbitrary mid-step recovery is not supported.

`on_checkpoint(output, checkpoint)` receives two `Path` values. It runs on the
training thread with training paused, after the checkpoint, log, references,
pruning, and any completion summary are durable. It must not change run files
or RNG state. Callback errors propagate; the local resume checkpoint is already
published. The campaign caller selects 600-second backups and a completion
backup. See [persistent run backups](persistence.md) for verified Drive copies,
restore checks, and the separate two-generation backup retention policy.

Keep active runs on the local Colab SSD, not Drive. Check disk capacity before
training. The bounded smoke does not establish full-campaign storage needs.

When publication is requested, commit the GAN fork first, the POG repository
second, and the parent repository last. Each parent must point to an available
child revision. Do not commit or push as part of training or setup.
Saving the notebook to Colab/Drive is not publication. Exclude the external
OpenSpec planning folder from publication.
