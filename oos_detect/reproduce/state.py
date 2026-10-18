"""Local run manifests and checkpoints at completed optimization boundaries."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile

import numpy as np
import torch

from .runtime import require_cuda


STREAMS = ("prepare", "initialization", "loader", "perturbation", "generation", "evaluation")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Write a new JSON file; never replace an existing artifact."""
    with Path(path).open("x", encoding="utf-8") as target:
        target.write(canonical_json(value) + "\n")


def derived_seed(seed, stream):
    if stream not in STREAMS:
        raise ValueError(f"Unknown RNG stream: {stream}")
    return int.from_bytes(hashlib.sha256(f"{seed}:{stream}".encode()).digest()[:8], "big") % (2**63 - 1)


def seed_run(seed, device="cuda:0", deterministic=False):
    selected = require_cuda(device)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    return {name: torch.Generator(device=selected).manual_seed(derived_seed(seed, name))
            for name in STREAMS}


def source_revisions(repository):
    root = Path(repository).resolve()
    pog = root / "oos_detect/models/pseudo-ood-generation"
    paths = {"root": root, "pog": pog,
             "gan": pog / "pseudo_ood_generation/components/wgan-gp"}
    result = {}
    for name, path in paths.items():
        def git(*args):
            return subprocess.run(["git", *args], cwd=path, check=True,
                                  capture_output=True, text=True).stdout.strip()
        result[name] = {"revision": git("rev-parse", "HEAD"),
                        "dirty": bool(git("status", "--porcelain"))}
    return result


def create_run(directory, config, data_hash, embedding_hash, repository, device="cuda:0"):
    """Create a new run after validating runtime and metadata; reject collisions."""
    selected = require_cuda(device)
    manifest = {
        "format_version": 1, "config": config, "data_hash": data_hash,
        "embedding_hash": embedding_hash, "source": source_revisions(repository),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": str(selected),
                    "gpu": torch.cuda.get_device_name(selected)},
        "seeds": {name: derived_seed(config["seed"], name) for name in STREAMS},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    manifest["compatibility_hash"] = fingerprint({
        "config": config, "data_hash": data_hash, "embedding_hash": embedding_hash,
    })
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "config.json", config)
    return manifest


def log_phase(directory, record):
    """Append finite JSON diagnostics without an online service."""
    line = canonical_json(record)
    with (Path(directory) / "phases.jsonl").open("a", encoding="utf-8") as output:
        output.write(line + "\n")


@dataclass(frozen=True)
class Boundary:
    """Position after a completed phase; next_phase is the next work to execute."""

    epoch: int
    batch: int
    next_phase: str

    def __post_init__(self):
        if self.epoch < 0 or self.batch < 0 or not self.next_phase:
            raise ValueError("A checkpoint requires a valid completed phase/epoch boundary.")


def save_checkpoint(path, *, models, optimizers, generators, boundary,
                    compatibility_hash, temperature, policy_baseline, extra=None):
    """Save all training state. The destination must not already exist."""
    if temperature <= 0:
        raise ValueError("Checkpoint temperature must be positive.")
    state = {
        "format_version": 1, "compatibility_hash": compatibility_hash,
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "streams": {name: generator.get_state() for name, generator in generators.items()}},
        "boundary": vars(boundary), "temperature": temperature,
        "policy_baseline": policy_baseline, "extra": extra or {},
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            torch.save(state, target)
            target.flush()
            os.fsync(target.fileno())
        # A hard link publishes the completed file without replacing another file.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def restore_checkpoint(path, *, models, optimizers, generators, compatibility_hash,
                       device="cuda:0"):
    """Restore a trusted local checkpoint, including loader streams and global RNG."""
    selected = require_cuda(device)
    # Full optimizer/Python RNG state needs pickle. Never load untrusted checkpoints.
    state = torch.load(path, map_location=selected, weights_only=False)
    if state.get("format_version") != 1 or state.get("compatibility_hash") != compatibility_hash:
        raise ValueError("Checkpoint data, embeddings, architecture, or configuration does not match this run.")
    for section, objects in (("models", models), ("optimizers", optimizers)):
        if set(state[section]) != set(objects):
            raise ValueError(f"Checkpoint {section} do not match this run.")
    if set(state["rng"]["streams"]) != set(generators):
        raise ValueError("Checkpoint RNG streams do not match this run.")
    boundary = Boundary(**state["boundary"])
    for name, model in models.items():
        model.load_state_dict(state["models"][name], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
    for name, generator in generators.items():
        generator.set_state(state["rng"]["streams"][name].cpu())
    return {"boundary": boundary, "temperature": state["temperature"],
            "policy_baseline": state["policy_baseline"], "extra": state["extra"]}
