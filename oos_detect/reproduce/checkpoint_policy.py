"""Operational checkpoint timing, durable logs, and owned-file retention."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time


def _atomic(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-policy-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconcile_logs(directory, stage, serial, last_record=None):
    """Remove only this stage's speculative records; repair a missing saved record.

    Keep unrelated lines byte-for-byte. Refuse malformed JSON rather than silently
    delete data. A checkpoint stores its last record to cover save-before-log crashes.
    """
    path = Path(directory) / "phases.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    kept, seen = [], set()
    for line in lines:
        record = json.loads(line)
        owned = ("boundary_serial" in record and record.get("phase") in {
                     "reconstruction", "auxiliary", "critic", "generator_wasserstein",
                     "generator_entropy", "dmix_reconstruction", "dmix_critic"} if stage == "generator"
                 else record.get("phase") == "classifier_epoch")
        position = record.get("boundary_serial") if stage == "generator" else record.get("epoch")
        if owned and isinstance(position, int):
            if position > serial or position in seen:
                continue
            seen.add(position)
        kept.append(line)
    if last_record is not None and serial not in seen:
        if kept and not kept[-1].endswith("\n"):
            kept[-1] += "\n"
        kept.append(json.dumps(last_record, sort_keys=True, allow_nan=False) + "\n")
    _atomic(path, "".join(kept))


class CheckpointPolicy:
    """Keep policy out of model compatibility. None preserves phase saves.

    ``on_checkpoint(output, checkpoint)`` receives two positional pathlib.Path values.
    It runs on the training thread, with training paused, after the checkpoint,
    phase log, references, bounded pruning, and completion summary are durable.
    It may copy these files. It must not mutate files or training RNG state.
    Exceptions propagate with the local resume checkpoint already published.
    """

    def __init__(self, directory, interval=None, retain=False, callback=None):
        if interval is not None and (isinstance(interval, bool) or not isinstance(interval, (int, float))
                                     or not math.isfinite(interval) or interval <= 0):
            raise ValueError("checkpoint_interval_seconds must be finite and positive, or None")
        if type(retain) is not bool:
            raise ValueError("retain_checkpoints must be a boolean")
        if callback is not None and not callable(callback):
            raise ValueError("on_checkpoint must be callable")
        self.directory = Path(directory)
        self.interval, self.retain, self.callback = interval, retain, callback
        self.last_saved = time.monotonic()
        self.path = self.directory / "checkpoint-policy.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"owned": {}, "references": {}}

    def due(self, *, force=False):
        return force or self.interval is None or time.monotonic() - self.last_saved >= self.interval

    def guard(self, previous=None, *, estimate=0):
        """Reserve a full new checkpoint plus 64 MiB or 10% headroom.

        Before the first save, reserve at least 256 MiB. The caller also supplies
        a conservative estimate from model tensors for lazy optimizer allocation.
        """
        size = Path(previous).stat().st_size if previous is not None else max(256 * 1024**2, estimate)
        required = size + max(64 * 1024**2, size // 10)
        free = shutil.disk_usage(self.directory).free
        if free < required:
            raise OSError(f"Insufficient checkpoint disk space: need {required} bytes; free {free}")
        log = self.directory / "phases.jsonl"
        if log.exists():
            with log.open("rb") as source:
                os.fsync(source.fileno())

    def publish(self, checkpoint, *, selected=None, final=None):
        checkpoint = Path(checkpoint)
        sync_directory(checkpoint.parent)
        log = self.directory / "phases.jsonl"
        if log.exists():
            with log.open("rb") as source:
                os.fsync(source.fileno())
        summary = self.directory / "summary.json"
        if summary.exists():
            with summary.open("rb") as source:
                os.fsync(source.fileno())
        self.state["owned"][checkpoint.name] = _digest(checkpoint)
        self.state["references"] = {key: Path(value).name for key, value in
                                    (("latest", checkpoint), ("selected", selected), ("final", final))
                                    if value is not None}
        _atomic(self.path, json.dumps(self.state, sort_keys=True) + "\n")
        self.last_saved = time.monotonic()

    def notify_and_prune(self, checkpoint):
        if self.retain:
            keep = set(self.state["references"].values())
            for name, digest in list(self.state["owned"].items()):
                if name in keep or re.fullmatch(r"boundary-\d{9,}\.pt", name) is None:
                    continue
                path = self.directory / "checkpoints" / name
                if path.is_file() and not path.is_symlink() and _digest(path) == digest:
                    path.unlink()
                    del self.state["owned"][name]
            sync_directory(Path(checkpoint).parent)
            _atomic(self.path, json.dumps(self.state, sort_keys=True) + "\n")
        if self.callback is not None:
            self.callback(self.directory, Path(checkpoint))


def model_checkpoint_estimate(models):
    """Allow weights, gradients, two Adam slots, and serialization overhead."""
    return 5 * sum(value.numel() * value.element_size()
                   for model in models.values() for value in model.state_dict().values())
