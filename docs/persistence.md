# Persistent run backups

Keep training files and checkpoints on the local Colab SSD. Do not set the run
directory to mounted Drive. The checkpoint writer uses `os.link`, which is for
the local filesystem. The backup helper rejects FUSE and known Drive paths.

Call `backup_run(run, backupdir, metadata=None)` on the training thread after a
completed checkpoint and its phase log writes. Do not change the run while this
call runs. The caller selects a 10-minute interval and calls it again at run
completion. The helper does not start a timer or a background thread.

Use a dedicated Drive backup directory for each run. The helper puts all regular
run files in a local `.tar.gz` archive with gzip level 1. This includes all retained
checkpoints, config, manifests, phase logs, results, and file hashes. It rejects
links and special files. Standard-library `tarfile` avoids another dependency and
supports explicit member checks without general-purpose archive extraction.

Compression retains all phase log records. It does not change local training
files or convert existing backup archives. Restore accepts both original `.tar`
archives and new `.tar.gz` archives. Both formats use the same hash checks and
completion manifests. The two-generation limit counts both formats together.

Each archive has a unique generation name. The helper copies it to Drive, closes
it, then reads it back to check the archive SHA256 and each file SHA256. Only then
does it write the `.complete.json` manifest. Drive publication does not use a
rename or a hard link. A partial copy or manifest is not a valid backup.

After verification, retain the newest two valid generations for that run. Delete
only recognized archive files with paired valid completion manifests. Leave
corrupt, incomplete, and unrelated files in place. These files can use extra
space. Do not delete a previous valid backup before a new backup is verified.
Use one caller per backup directory; concurrent backup writers are not supported.

## Restore

Call `restore_run(backupdir, destination, expected_metadata=None)`. The destination
must be a new local directory with an existing parent. The helper returns the
selected completion manifest, or `None` if no recognized completion manifest
exists. It raises an error if recognized generations exist but none are valid.

Restore first copies an archive to local storage. It checks the whole archive,
all member hashes, paths, and file types before it creates the destination. It
rejects traversal paths, symbolic links, hard links, duplicates, and extra files.
If the latest archive is corrupt, it tries the previous valid generation. A
restore I/O failure can leave a partial destination. Keep it for inspection and
use a different new destination for a retry. The helper does not remove user work.

Pass expected data and embedding hashes, source revisions, config, and runtime
from the caller's verified records. Each supplied top-level metadata value must
match exactly, including nested runtime or source values. A mismatch does not
change the archive or destination. Caller metadata cannot replace a value in the
run manifest. Missing provenance is not inferred or fabricated. A checksum checks
integrity, not identity: use only your trusted backup root. Checkpoints can contain
pickle data; this helper does not load them.

The restored directory contains the checkpoint set and phase logs from one
completed boundary. The training caller must resume its highest completed
checkpoint with those logs and the matching run manifest. Do not combine files
from different generations or invent missing runtime records.

Free-space checks reserve uncompressed space plus archive overhead for snapshot
creation. Restore reserves the stored archive size plus all restored file sizes.
Checks also reserve room for the Drive copy where filesystem capacity is
available. Drive `disk_usage` output is not a check of the account's global
storage quota. Upload and I/O errors can still occur. Failed copies do not remove
previous valid backups.

## References

- [Colab FAQ: Drive I/O and quotas](https://research.google.com/colaboratory/faq.html)
- [Python 3.12 tarfile](https://docs.python.org/3.12/library/tarfile.html)
- [Python 3.12 hashlib](https://docs.python.org/3.12/library/hashlib.html)
- [Python 3.12 shutil](https://docs.python.org/3.12/library/shutil.html)
