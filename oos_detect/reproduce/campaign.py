"""Run one paper experiment. Never advance to the next experiment."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from . import backup, pipeline
from .config import paper_config
from .data import ORIGINAL_DATA, file_hash, prepare_data


EXPERIMENTS = ("MSP", "Perturb", "Mix", "POG", "AEPOG",
               "pog-no-noise", "pog-no-soft-token")
SEEDS = tuple(range(10))
ORDER = tuple((experiment, seed) for seed in SEEDS for experiment in EXPERIMENTS)
FORMAT = "oos-detect-paper-campaign-v1"


def next_requested(experiment, seed):
    """Return the next request in seed-major order. Do not run it."""
    index = ORDER.index((experiment, seed)) + 1
    if index == len(ORDER):
        return None
    method, next_seed = ORDER[index]
    return {"experiment": method, "seed": next_seed}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    # Exclusive publication also protects unknown files after interruption.
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _log(run, event, **values):
    with (run / "campaign.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **values}, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collect_identity(embeddings, device):
    """Require clean source and record the runtime before any resume."""
    import torch
    from .runtime import require_cuda
    from .state import source_revisions

    repository = pipeline.REPOSITORY
    source = source_revisions(repository)
    dataset = repository / "oos_detect/datasets/oos_eval"
    def git_dataset(*arguments):
        return subprocess.run(["git", *arguments], cwd=dataset, check=True,
                              capture_output=True, text=True).stdout.strip()

    source["dataset"] = {"revision": git_dataset("rev-parse", "HEAD"),
                         "dirty": bool(git_dataset("status", "--porcelain"))}
    if any(item["dirty"] for item in source.values()):
        raise ValueError("Paper campaigns require a clean root and clean submodules.")
    modules = subprocess.run(["git", "submodule", "status", "--recursive"],
                             cwd=repository, check=True, capture_output=True,
                             text=True).stdout.splitlines()
    if any(not line.startswith(" ") for line in modules):
        raise ValueError("Initialize submodules at their pinned revisions before training.")
    selected = require_cuda(device)
    return {
        "source": source, "submodules": [
            {"revision": line[1:].split()[0], "path": line[1:].split()[1]}
            for line in modules],
        "uv_lock_sha256": file_hash(repository / "uv.lock"),
        "embedding_sha256": file_hash(embeddings),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "cuda": torch.version.cuda, "device": str(selected),
                    "gpu": torch.cuda.get_device_name(selected),
                    "compute_capability": list(torch.cuda.get_device_capability(selected)),
                    "cudnn": torch.backends.cudnn.version()},
        "data_sources": {name: file_hash(ORIGINAL_DATA / name)
                         for name in ("data_full.json", "data_oos_plus.json")},
    }


class _Backups:
    """Back up only while the synchronous caller has stopped training."""

    def __init__(self, run, destination, metadata, seconds, config):
        self.run, self.destination, self.metadata = run, destination, metadata
        self.seconds, self.config = seconds, config
        self.last = None
        self.seen = set()
        self.progress = {}

    def force(self, reason):
        started = time.monotonic()
        try:
            manifest = backup.backup_run(self.run, self.destination, metadata=self.metadata)
        except Exception:
            _log(self.run, "backup_failed", reason=reason,
                 seconds=time.monotonic() - started)
            raise
        self.last = time.monotonic()
        _log(self.run, "backup_verified", reason=reason, seconds=self.last - started,
             bytes=sum(item["size"] for item in manifest["files"].values()),
             generation=manifest["generation"])
        return manifest

    def _epochs(self, output, serial):
        if output.name != "generator":
            return float(serial), self.config.classifier_epochs
        prepared = _read(self.run / "prepared/manifest.json")
        batches = math.ceil(prepared["splits"]["train"]["count"] / self.config.batch_size)
        return serial / (batches * len(pipeline.generator_phases(self.config))), self.config.generator_epochs

    def start_stage(self, output, resume=None):
        serial = 0 if resume is None else int(Path(resume).stem.removeprefix("boundary-"))
        epoch, _ = self._epochs(output, serial)
        self.progress[output] = (time.monotonic(), epoch)

    def checkpoint(self, output, checkpoint):
        output, checkpoint = Path(output), Path(checkpoint)
        if output.parent != self.run or checkpoint.parent != output / "checkpoints":
            raise ValueError("Checkpoint callback is outside the controlled run.")
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError("Checkpoint callback requires a durable regular checkpoint.")
        serial = int(checkpoint.stem.removeprefix("boundary-"))
        epochs, total = self._epochs(output, serial)
        now = time.monotonic()
        previous_time, previous_epoch = self.progress.get(output, (now, None))
        elapsed = now - previous_time
        completed = None if previous_epoch is None else epochs - previous_epoch
        per_epoch = elapsed / completed if completed and completed > 0 else None
        # This wall time includes checkpoint I/O and validation, not just updates.
        _log(self.run, "checkpoint", stage=output.name, checkpoint=checkpoint.name,
             checkpoint_bytes=checkpoint.stat().st_size, completed_epochs=epochs,
             wall_seconds_since_checkpoint=elapsed, seconds_per_epoch=per_epoch,
             remaining_seconds=None if per_epoch is None else max(0, total - epochs) * per_epoch)
        if output not in self.seen or self.last is None or now - self.last >= self.seconds:
            self.force("checkpoint:" + output.name)
            self.seen.add(output)
        # Include backup time in the next observed wall-throughput measurement.
        self.progress[output] = (now, epochs)


def _training_resume(output, epochs):
    if not output.exists():
        return False, None
    summary = output / "summary.json"
    if summary.is_file():
        result = _read(summary)
        if result.get("complete") is True and result.get("completed_epochs") == epochs:
            checkpoint = output / "checkpoints" / Path(result["checkpoint"]).name
            if checkpoint.is_file() and not checkpoint.is_symlink():
                return True, None
            raise ValueError(f"Completed stage is missing its checkpoint: {output}")
    candidates = sorted((output / "checkpoints").glob("boundary-*.pt"))
    if not candidates or any(path.is_symlink() for path in candidates):
        raise ValueError(f"Partial stage has no safe durable checkpoint: {output}; preserve it and use a new work root.")
    return False, candidates[-1]


def _reject_links(run):
    if run.is_symlink() or any(path.is_symlink() for path in run.rglob("*")):
        raise ValueError("Controlled runs must not contain symlinks.")


def _run_experiment(*, experiment="MSP", seed=0, work_root, backup_root, embeddings,
                   backup_seconds=600, checkpoint_seconds=60, device="cuda:0"):
    """Prepare, resume, evaluate, and back up exactly one method and seed."""
    if experiment not in EXPERIMENTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Select one supported experiment and a seed from 0 through 9.")
    for name, value in (("backup_seconds", backup_seconds), ("checkpoint_seconds", checkpoint_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    # Reuse the backup worker's mount checks; hard-linked checkpoints cannot use Drive.
    work = backup._local(work_root)
    run = work / experiment / f"seed-{seed}"
    destination = Path(backup_root).resolve() / experiment / f"seed-{seed}"
    if destination == run or run in destination.parents or destination in run.parents:
        raise ValueError("Work and backup directories must not overlap.")
    embeddings = Path(embeddings).resolve(strict=True)
    variant = "POG" if experiment.startswith("pog-no-") else experiment
    config = paper_config(variant, mode=experiment if experiment.startswith("pog-no-") else None,
                          device=device)
    metadata = {"experiment": experiment, "seed": seed, "config": config.for_seed(seed),
                **collect_identity(embeddings, device)}
    ownership = {"format": FORMAT, **metadata}
    if run.parent.is_symlink():
        raise ValueError("Controlled experiment directory must not be a symlink.")
    run.parent.mkdir(parents=True, exist_ok=True)
    if run.is_symlink():
        raise ValueError("Controlled run must not be a symlink.")
    if not run.exists():
        restored = backup.restore_run(destination, run, expected_metadata=ownership)
        if restored is None:
            run.mkdir(exist_ok=False)
            _write(run / "manifest.json", ownership)
    _reject_links(run)
    if not (run / "manifest.json").is_file() or _read(run / "manifest.json") != ownership:
        raise ValueError("Existing run ownership, source, embeddings, or runtime mismatch; no files were discarded.")
    snapshots = _Backups(run, destination, metadata, backup_seconds, config)
    prepared = run / "prepared"
    if prepared.exists():
        loaded = pipeline.load_prepared(prepared, config)
        if loaded.manifest.get("seed") != seed:
            raise ValueError("Prepared Dmix seed differs from the requested seed.")
        expected_sources = metadata["data_sources"]
        if any(expected_sources.get(item["file"]) != item["sha256"]
               for item in loaded.manifest["sources"].values()):
            raise ValueError("Prepared data source identity differs from the original CLINC sources.")
    else:
        prepare_data(prepared, seed=seed, max_length=config.max_length,
                     profile=config.vocabulary_profile, ind_count=config.dmix_ind_count)
    snapshots.force("prepared-complete")
    common = {"prepared": prepared, "config": config, "seed": seed}

    def train(name, operation, epochs, **inputs):
        output = run / name
        complete, resume = _training_resume(output, epochs)
        if not complete:
            snapshots.start_stage(output, resume)
            result = operation(**common, output=output, embedding_path=embeddings,
                               resume=resume, checkpoint_interval_seconds=checkpoint_seconds,
                               retain_checkpoints=True, on_checkpoint=snapshots.checkpoint, **inputs)
            if result.get("complete") is not True:
                raise ValueError(f"Training stopped before stage completion: {name}")
            complete, _ = _training_resume(output, epochs)
            if not complete:
                raise ValueError(f"Training did not publish its completion summary: {name}")
        snapshots.force(name + "-complete")

    generator = generated = None
    if config.uses_generator:
        generator, generated = run / "generator", run / "generated"
        train("generator", pipeline.train_generator, config.generator_epochs)
        if generated.exists():
            if (not (generated / "manifest.json").is_file()
                    or not (generated / "config.json").is_file()
                    or _read(generated / "config.json") != config.for_seed(seed)
                    or not (generated / "generator-run-manifest.json").is_file()):
                raise ValueError("Partial generated export cannot be resumed; preserve it and use a new work root.")
        else:
            pipeline.generate(**common, generator_run=generator, output=generated)
        snapshots.force("generated-complete")
    train("classifier", pipeline.train_classifier, config.classifier_epochs,
          generated=generated, generator_run=generator)
    evaluation = run / "evaluation"
    if evaluation.exists():
        if (not (evaluation / "summary.json").is_file()
                or not (evaluation / "config.json").is_file()
                or _read(evaluation / "config.json") != config.for_seed(seed)
                or not (evaluation / "run-manifest.json").is_file()):
            raise ValueError("Partial evaluation cannot be resumed; preserve it and use a new work root.")
        report = _read(evaluation / "summary.json")
    else:
        report = pipeline.evaluate(**common, classifier_run=run / "classifier", output=evaluation)
    snapshots.force("evaluation-complete")
    result = {"complete": True, "experiment": experiment, "seed": seed,
              "run": str(run), "evaluation": report,
              "next_requested": next_requested(experiment, seed),
              "auto_advance": False}
    completion = run / "completed.json"
    # Local paths change after a verified restore to a replacement SSD.
    completed_record = {key: value for key, value in result.items() if key != "run"}
    if completion.exists():
        if _read(completion) != completed_record:
            raise ValueError("Existing campaign completion differs; no files were overwritten.")
    else:
        _write(completion, completed_record)
    snapshots.force("campaign-complete")
    return result


def run_experiment(*, experiment="MSP", seed=0, work_root, backup_root, embeddings,
                   backup_seconds=600, checkpoint_seconds=60, device="cuda:0"):
    """Refuse a second writer for the same local experiment and seed."""
    if experiment not in EXPERIMENTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Select one supported experiment and a seed from 0 through 9.")
    for name, value in (("backup_seconds", backup_seconds), ("checkpoint_seconds", checkpoint_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    work = backup._local(work_root)
    work.mkdir(parents=True, exist_ok=True)
    lock_path = work / f".campaign-{experiment}-seed-{seed}.lock"
    if lock_path.is_symlink():
        raise ValueError("Campaign lock must not be a symlink.")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This experiment and seed already have an active local writer.") from error
        try:
            return _run_experiment(
                experiment=experiment, seed=seed, work_root=work,
                backup_root=backup_root, embeddings=embeddings,
                backup_seconds=backup_seconds, checkpoint_seconds=checkpoint_seconds,
                device=device,
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one paper experiment; never auto-advance")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="MSP")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--backup-seconds", type=float, default=600)
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    parser.add_argument("--device", default="cuda:0")
    result = run_experiment(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
