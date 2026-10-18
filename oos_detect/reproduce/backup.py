"""Synchronous, verified run snapshots; no training or CUDA imports."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
import time
import uuid
import zlib


_FORMAT = "oos-detect-backup-v1"
_GENERATION = re.compile(r"oos-backup-([0-9a-f]{64})-([0-9]{20})-([0-9a-f]{32})")
_HASH = re.compile(r"[0-9a-f]{64}")
_RESERVE = 1024 * 1024
_INVALID = (OSError, ValueError, KeyError, TypeError, EOFError, tarfile.TarError, zlib.error)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _local(path):
    """Reject known Drive paths and network/FUSE mounts on Linux."""
    path = Path(path).resolve()
    if path == Path("/content/drive") or Path("/content/drive") in path.parents:
        raise ValueError("Training, snapshots, and restore destinations must be local, not Drive.")
    mounts = Path("/proc/self/mountinfo")
    if mounts.exists():
        selected = (0, "")
        for line in mounts.read_text().splitlines():
            fields, filesystem = line.split(" - ", 1)
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields.split()[4]))
            if (path == mount or mount in path.parents) and len(str(mount)) > selected[0]:
                selected = (len(str(mount)), filesystem.split()[0])
        if selected[1].startswith(("fuse", "nfs", "cifs", "smb")):
            raise ValueError("Training, snapshots, and restore destinations require a local filesystem.")
    return path


def _space(path, required):
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return  # Drive may not expose filesystem capacity. This is not a quota check.
    if free < required + _RESERVE:
        raise OSError(f"Insufficient reported free space at {path}: need {required + _RESERVE} bytes.")


def _safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or name == "." or ":" in path.parts[0]):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _inventory(run):
    files = {}
    for path in sorted(run.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Run contains a symlink or special file: {path}")
        name = path.relative_to(run).as_posix()
        _safe_name(name)
        files[name] = {"size": path.stat().st_size, "sha256": _hash(path)}
    if "manifest.json" not in files:
        raise ValueError("Run must contain manifest.json.")
    return files


def _verify(archive, manifest):
    if archive.is_symlink() or not archive.is_file() or _hash(archive) != manifest["archive_sha256"]:
        raise ValueError("Archive SHA256 mismatch.")
    expected = manifest["files"]
    if not isinstance(expected, dict) or "manifest.json" not in expected:
        raise ValueError("Missing per-file manifest.")
    for name, record in expected.items():
        _safe_name(name)
        if (not isinstance(record, dict) or type(record.get("size")) is not int
                or record["size"] < 0 or not _HASH.fullmatch(record.get("sha256", ""))):
            raise ValueError("Invalid per-file manifest.")
    seen = set()
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            _safe_name(member.name)
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise ValueError("Archive contains a link, special file, duplicate, or unexpected member.")
            record = expected[member.name]
            if member.size != record["size"]:
                raise ValueError("Archive member size mismatch.")
            with source.extractfile(member) as stream:
                if member.name == "manifest.json":
                    # Read once: a backward gzip seek would decompress preceding checkpoints again.
                    content = stream.read()
                    digest = hashlib.sha256(content).hexdigest()
                else:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != record["sha256"]:
                    raise ValueError("Archive member SHA256 mismatch.")
            if member.name == "manifest.json":
                recorded = json.loads(content)
                if not isinstance(recorded, dict) or any(
                        key not in manifest["metadata"] or manifest["metadata"][key] != value
                        for key, value in recorded.items()):
                    raise ValueError("Backup metadata conflicts with the archived run manifest.")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive is missing files.")


def _candidates(root):
    for path in sorted(root.glob("oos-backup-*.complete.json"), reverse=True):
        generation = path.name.removesuffix(".complete.json")
        match = _GENERATION.fullmatch(generation)
        if not match or path.is_symlink() or not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (manifest["format"] != _FORMAT or manifest["generation"] != generation
                    or manifest["run_id"] != match[1]
                    or manifest["archive"] not in (generation + ".tar", generation + ".tar.gz")
                    or not _HASH.fullmatch(manifest["archive_sha256"])
                    or not isinstance(manifest["metadata"], dict)):
                continue
            yield path, manifest
        except (OSError, ValueError, KeyError, TypeError):
            continue


def _copy(source, destination):
    # Exclusive creation; never rename or hard-link files on Drive.
    with Path(source).open("rb") as incoming, Path(destination).open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()


def backup_run(run, backupdir, metadata=None):
    """Snapshot a quiescent local run on the training thread, then verify Drive."""
    run = _local(run)
    root = Path(backupdir).resolve()
    if root == run or run in root.parents:
        raise ValueError("Backup root must be outside the run directory.")
    files = _inventory(run)
    provenance = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("Run manifest must be a JSON object.")
    # Never replace recorded provenance with caller-supplied claims.
    for key, value in (metadata or {}).items():
        if key in provenance and provenance[key] != value:
            raise ValueError(f"Metadata conflicts with run manifest: {key}")
        provenance[key] = value
    provenance = json.loads(_json(provenance))
    run_id = hashlib.sha256(str(run).encode()).hexdigest()
    generation = f"oos-backup-{run_id}-{time.time_ns():020d}-{uuid.uuid4().hex}"
    # Reserve uncompressed space, including headers, even for compressible logs.
    required = sum(record["size"] + 4096 + len(name.encode()) * 2
                   for name, record in files.items()) + 10240
    required += required // 1000 + 4096  # Allow gzip overhead for incompressible data.
    _space(run.parent, required)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oos-snapshot-", dir=run.parent) as temporary:
        archive = Path(temporary) / (generation + ".tar.gz")
        with tarfile.open(archive, "x:gz", compresslevel=1, dereference=True) as target:
            for name in files:
                target.add(run / name, arcname=name, recursive=False)
        manifest = {"format": _FORMAT, "generation": generation, "run_id": run_id,
                    "archive": archive.name, "archive_sha256": _hash(archive),
                    "files": files, "metadata": provenance}
        _verify(archive, manifest)
        if _inventory(run) != files:
            raise ValueError("Run changed during snapshot; use the completed-checkpoint training callback.")
        _space(root, archive.stat().st_size + len(_json(manifest).encode()))
        remote = root / archive.name
        _copy(archive, remote)
        _verify(remote, manifest)
        completion = root / (generation + ".complete.json")
        with completion.open("x", encoding="utf-8") as output:
            output.write(_json(manifest) + "\n")
        if json.loads(completion.read_text(encoding="utf-8")) != manifest:
            raise OSError("Completion manifest read-back failed.")
    # Bound only our verified, paired generations, separately for each run.
    verified = []
    for completion, candidate in _candidates(root):
        if candidate["run_id"] != run_id:
            continue
        try:
            _verify(root / candidate["archive"], candidate)
        except _INVALID:
            continue
        verified.append((completion, candidate))
    for completion, candidate in verified[2:]:
        # Remove the completion marker first; an interruption leaves an ignored archive.
        completion.unlink()
        (root / candidate["archive"]).unlink()
    return manifest


def restore_run(backupdir, destination, expected_metadata=None):
    """Restore the newest valid snapshot to a new local directory; do not load pickle."""
    if os.path.lexists(destination):
        raise FileExistsError(f"Restore destination already exists: {destination}")
    destination = _local(destination)
    if destination.exists():
        raise FileExistsError(f"Restore destination already exists: {destination}")
    root = Path(backupdir)
    candidates = sorted(_candidates(root), key=lambda item: item[1]["generation"].split("-")[-2],
                        reverse=True)
    if not candidates:
        return None
    if not destination.parent.is_dir():
        raise FileNotFoundError("Restore parent must already exist.")
    mismatch = False
    for _, manifest in candidates:
        if any(key not in manifest["metadata"] or manifest["metadata"][key] != value
               for key, value in (expected_metadata or {}).items()):
            mismatch = True
            continue
        try:
            archive = root / manifest["archive"]
            if archive.is_symlink():
                continue
            required = archive.stat().st_size + sum(record["size"] for record in manifest["files"].values())
            _space(destination.parent, required)
            with tempfile.TemporaryDirectory(prefix=".oos-restore-", dir=destination.parent) as temporary:
                local = Path(temporary) / manifest["archive"]
                _copy(archive, local)
                _verify(local, manifest)
                # All validation precedes destination creation. No extractall or links.
                destination.mkdir(exist_ok=False)
                with tarfile.open(local, "r:*") as source:
                    for member in source:
                        target = destination.joinpath(*PurePosixPath(member.name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as incoming, target.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                return manifest
        except _INVALID:
            if destination.exists():
                raise  # Never overwrite or remove a partially restored destination.
            continue
    if mismatch:
        raise ValueError("Backup metadata does not match the expected provenance or runtime.")
    raise ValueError("No valid backup generation remains.")
