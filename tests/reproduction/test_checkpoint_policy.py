"""File-only policy tests. Model integration checks are in test_pipeline."""

import json
from types import SimpleNamespace

import pytest

from oos_detect.reproduce import checkpoint_policy as policy


def checkpoint(tmp_path, serial):
    directory = tmp_path / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"boundary-{serial:09d}.pt"
    path.write_bytes(f"owned checkpoint {serial}".encode())
    return path


def test_interval_and_forced_boundaries(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(policy.time, "monotonic", lambda: now[0])
    bounded = policy.CheckpointPolicy(tmp_path, 60)
    assert not bounded.due()
    now[0] += 59
    assert not bounded.due()
    assert bounded.due(force=True)
    now[0] += 1
    assert bounded.due()
    saved = checkpoint(tmp_path, 1)
    bounded.publish(saved)
    assert not bounded.due()
    assert policy.CheckpointPolicy(tmp_path).due()


@pytest.mark.parametrize("interval", [0, -1, True, "60", float("inf"), float("nan")])
def test_invalid_intervals(tmp_path, interval):
    with pytest.raises(ValueError, match="interval"):
        policy.CheckpointPolicy(tmp_path, interval)


def test_pruning_is_owned_hash_checked_and_precedes_callback(tmp_path):
    seen = []

    def callback(output, current):
        references = json.loads((output / "checkpoint-policy.json").read_text())["references"]
        assert current.is_file()
        assert references["latest"] == current.name
        seen.append({path.name for path in current.parent.iterdir()})

    bounded = policy.CheckpointPolicy(tmp_path, retain=True, callback=callback)
    first, selected, changed, unrelated = [checkpoint(tmp_path, serial) for serial in range(1, 5)]
    for path in (first, selected, changed):
        bounded.publish(path)
    changed.write_bytes(b"user replacement")
    current = checkpoint(tmp_path, 5)
    bounded.publish(current, selected=selected, final=current)
    bounded.notify_and_prune(current)
    assert not first.exists()
    assert selected.exists() and changed.exists() and unrelated.exists()
    assert first.name not in seen[0]
    resumed = policy.CheckpointPolicy(tmp_path, retain=True)
    latest = checkpoint(tmp_path, 6)
    resumed.publish(latest, selected=selected, final=current)
    resumed.notify_and_prune(latest)
    assert current.exists() and selected.exists()


def test_default_retains_all_files_and_callback_errors_propagate(tmp_path):
    first, second = checkpoint(tmp_path, 1), checkpoint(tmp_path, 2)
    bounded = policy.CheckpointPolicy(tmp_path)
    bounded.publish(first)
    bounded.publish(second)
    bounded.notify_and_prune(second)
    assert first.exists()

    def fail(*args):
        raise RuntimeError("backup failed")

    bounded.callback = fail
    with pytest.raises(RuntimeError, match="backup failed"):
        bounded.notify_and_prune(second)
    assert second.exists()


@pytest.mark.parametrize("stage,key,phase", [("generator", "boundary_serial", "critic"),
                                           ("classifier", "epoch", "classifier_epoch")])
def test_reconcile_speculation_duplicate_and_checkpoint_before_log(tmp_path, stage, key, phase):
    unrelated = '{"phase":"user-note","boundary_serial":999,"epoch":999}\n'
    rows = [{"phase": phase, key: serial} for serial in (1, 1, 3, 4)]
    path = tmp_path / "phases.jsonl"
    path.write_text(unrelated + "".join(json.dumps(row) + "\n" for row in rows))
    last = {"phase": phase, key: 2, "loss": 0.25}
    policy.reconcile_logs(tmp_path, stage, 2, last)
    assert path.read_text().startswith(unrelated)
    result = [json.loads(line) for line in path.read_text().splitlines()]
    assert result[1:] == [rows[0], last]
    before = path.read_bytes()
    policy.reconcile_logs(tmp_path, stage, 2, last)
    assert path.read_bytes() == before


def test_reconcile_refuses_malformed_data_without_replacing_it(tmp_path):
    path = tmp_path / "phases.jsonl"
    path.write_text("not JSON\n")
    with pytest.raises(json.JSONDecodeError):
        policy.reconcile_logs(tmp_path, "generator", 1)
    assert path.read_text() == "not JSON\n"


def test_space_guard_first_and_previous_size(tmp_path, monkeypatch):
    bounded = policy.CheckpointPolicy(tmp_path)
    monkeypatch.setattr(policy.shutil, "disk_usage", lambda path: SimpleNamespace(free=300 * 1024**2))
    with pytest.raises(OSError, match="Insufficient checkpoint disk space"):
        bounded.guard()
    previous = checkpoint(tmp_path, 1)
    bounded.guard(previous)
    with pytest.raises(OSError, match="Insufficient checkpoint disk space"):
        bounded.guard(estimate=400 * 1024**2)


def test_no_durable_boundary_preserves_interrupted_run(tmp_path):
    from oos_detect.reproduce.pipeline import _new

    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="Output already exists"):
        _new(tmp_path)
    assert (tmp_path / "manifest.json").read_text() == "{}"
