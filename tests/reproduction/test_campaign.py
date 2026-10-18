"""Campaign I/O orchestration. Execute these tests only in the Colab runtime."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oos_detect.reproduce import campaign


@pytest.fixture
def orchestration(monkeypatch, tmp_path):
    calls = []
    identity = {"source": {"root": {"revision": "clean", "dirty": False}},
                "submodules": [" pinned dataset"], "uv_lock_sha256": "a" * 64,
                "embedding_sha256": "b" * 64, "runtime": {"cuda": "test-cuda"},
                "data_sources": {"data_full.json": "c" * 64,
                                 "data_oos_plus.json": "d" * 64}}
    monkeypatch.setattr(campaign, "collect_identity", lambda *args: identity)
    embeddings = tmp_path / "glove.6B.100d.txt"
    embeddings.write_text("supplied embeddings", encoding="utf-8")
    arguments = {"work_root": tmp_path / "ssd", "backup_root": tmp_path / "drive",
                 "embeddings": embeddings}

    def prepare(output, **kwargs):
        calls.append(("prepare", kwargs))
        output.mkdir()
        campaign._write(output / "manifest.json", {
            "seed": kwargs["seed"], "splits": {"train": {"count": 15000}},
            "sources": {key: {"file": key, "sha256": value}
                        for key, value in identity["data_sources"].items()}})

    def train(**kwargs):
        output = kwargs["output"]
        calls.append((output.name, kwargs))
        config = kwargs["config"]
        epochs = config.generator_epochs if output.name == "generator" else config.classifier_epochs
        output.mkdir(exist_ok=kwargs["resume"] is not None)
        (output / "checkpoints").mkdir(exist_ok=True)
        checkpoint = output / "checkpoints" / f"boundary-{epochs:09d}.pt"
        checkpoint.write_bytes(b"not a model: I/O fixture")
        campaign._write(output / "summary.json", {
            "complete": True, "completed_epochs": epochs, "checkpoint": str(checkpoint)})
        kwargs["on_checkpoint"](output, checkpoint)
        return {"complete": True}

    def generate(**kwargs):
        calls.append(("generate", kwargs))
        output = kwargs["output"]
        output.mkdir()
        for name, value in (("manifest.json", {}), ("generator-run-manifest.json", {}),
                            ("config.json", kwargs["config"].for_seed(kwargs["seed"]))):
            campaign._write(output / name, value)
        return {}

    def evaluate(**kwargs):
        calls.append(("evaluate", kwargs))
        output = kwargs["output"]
        output.mkdir()
        report = {"metrics": {"accuracy": 0.5}}
        for name, value in (("summary.json", report), ("run-manifest.json", {}),
                            ("config.json", kwargs["config"].for_seed(kwargs["seed"]))):
            campaign._write(output / name, value)
        return report

    monkeypatch.setattr(campaign, "prepare_data", prepare)
    monkeypatch.setattr(campaign.pipeline, "load_prepared", lambda path, config:
                        SimpleNamespace(manifest=campaign._read(path / "manifest.json")))
    monkeypatch.setattr(campaign.pipeline, "train_generator", train)
    monkeypatch.setattr(campaign.pipeline, "train_classifier", train)
    monkeypatch.setattr(campaign.pipeline, "generate", generate)
    monkeypatch.setattr(campaign.pipeline, "evaluate", evaluate)
    return SimpleNamespace(arguments=arguments, calls=calls, identity=identity)


def test_order_is_seed_major_and_never_launches_next():
    assert len(campaign.ORDER) == 70
    assert campaign.ORDER[:7] == tuple((name, 0) for name in campaign.EXPERIMENTS)
    assert campaign.next_requested("MSP", 0) == {"experiment": "Perturb", "seed": 0}
    assert campaign.next_requested("pog-no-soft-token", 0) == {"experiment": "MSP", "seed": 1}
    assert campaign.next_requested("pog-no-soft-token", 9) is None


def test_duplicate_local_writer_is_rejected(orchestration):
    import fcntl

    work = orchestration.arguments["work_root"]
    work.mkdir()
    with (work / ".campaign-MSP-seed-0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active local writer"):
            campaign.run_experiment(**orchestration.arguments)
    assert orchestration.calls == []


@pytest.mark.parametrize("experiment", campaign.EXPERIMENTS)
def test_one_full_paper_run(orchestration, experiment):
    result = campaign.run_experiment(**orchestration.arguments, experiment=experiment, seed=3)
    assert result["complete"] is True and result["auto_advance"] is False
    assert result["experiment"] == experiment and result["seed"] == 3
    calls = orchestration.calls
    uses_generator = experiment in ("POG", "AEPOG", "pog-no-noise", "pog-no-soft-token")
    assert [name for name, _ in calls] == (["prepare", "generator", "generate", "classifier", "evaluate"]
                                        if uses_generator else ["prepare", "classifier", "evaluate"])
    assert calls[0][1] == {"seed": 3, "max_length": 64, "profile": "paper", "ind_count": 10000}
    for name, kwargs in calls:
        if name not in ("generator", "classifier"):
            continue
        config = kwargs["config"]
        assert config.seeds == tuple(range(10))
        assert config.generator_epochs == 80 and config.classifier_epochs == 30
        assert config.embedding_dim == config.code_dim == config.noise_dim == 100
        assert config.random_only is False and kwargs["seed"] == 3
        assert kwargs["checkpoint_interval_seconds"] == 60 and kwargs["retain_checkpoints"] is True
        assert kwargs["embedding_path"] == orchestration.arguments["embeddings"]
        if experiment.startswith("pog-no-"):
            assert config.variant == "POG" and config.mode == experiment
        elif experiment == "AEPOG":
            assert config.variant == "AEPOG" and config.mode == "aepog"
    run = Path(result["run"])
    backupdir = orchestration.arguments["backup_root"] / experiment / "seed-3"
    manifests = list(backupdir.glob("*.complete.json"))
    assert len(manifests) == 2
    final = max((campaign._read(path) for path in manifests), key=lambda item: item["generation"])
    assert {"manifest.json", "prepared/manifest.json", "classifier/summary.json",
            "evaluation/summary.json", "completed.json"} <= set(final["files"])
    assert final["metadata"]["runtime"] == orchestration.identity["runtime"]
    assert final["metadata"]["embedding_sha256"] == "b" * 64
    logs = [json.loads(line) for line in (run / "campaign.jsonl").read_text().splitlines()]
    assert logs[-1]["reason"] == "campaign-complete" and logs[-1]["event"] == "backup_verified"
    assert all(row["bytes"] > 0 and row["seconds"] >= 0
               for row in logs if row["event"] == "backup_verified")


def test_completed_stages_are_not_retrained(orchestration):
    first = campaign.run_experiment(**orchestration.arguments)
    orchestration.calls.clear()
    assert campaign.run_experiment(**orchestration.arguments) == first
    assert orchestration.calls == []


@pytest.mark.parametrize("key,value", [("runtime", {"cuda": "changed"}),
                                       ("embedding_sha256", "changed"),
                                       ("source", {"root": {"revision": "changed"}}),
                                       ("uv_lock_sha256", "changed")])
def test_changed_identity_refuses_local_resume(orchestration, key, value):
    result = campaign.run_experiment(**orchestration.arguments)
    before = (Path(result["run"]) / "manifest.json").read_bytes()
    orchestration.identity[key] = value
    orchestration.calls.clear()
    with pytest.raises(ValueError, match="ownership, source, embeddings, or runtime mismatch"):
        campaign.run_experiment(**orchestration.arguments)
    assert orchestration.calls == []
    assert (Path(result["run"]) / "manifest.json").read_bytes() == before


def test_unknown_local_work_is_never_discarded(orchestration):
    run = orchestration.arguments["work_root"] / "MSP/seed-0"
    run.mkdir(parents=True)
    note = run / "user-work.txt"
    note.write_text("keep me")
    with pytest.raises(ValueError, match="ownership"):
        campaign.run_experiment(**orchestration.arguments)
    assert note.read_text() == "keep me" and list(run.iterdir()) == [note]
    assert orchestration.calls == []


def test_restore_uses_absent_destination_and_skips_completed_stages(orchestration, tmp_path):
    first = campaign.run_experiment(**orchestration.arguments, experiment="POG")
    orchestration.calls.clear()
    arguments = {**orchestration.arguments, "work_root": tmp_path / "replacement-ssd"}
    restored = campaign.run_experiment(**arguments, experiment="POG")
    assert orchestration.calls == []
    assert restored["evaluation"] == first["evaluation"]
    assert (Path(restored["run"]) / "classifier/summary.json").is_file()


def test_partial_training_resumes_latest_checkpoint(orchestration, monkeypatch):
    operation = campaign.pipeline.train_classifier

    def interrupt(**kwargs):
        output = kwargs["output"]
        (output / "checkpoints").mkdir(parents=True)
        (output / "checkpoints/boundary-000000001.pt").write_bytes(b"first")
        latest = output / "checkpoints/boundary-000000002.pt"
        latest.write_bytes(b"second")
        kwargs["on_checkpoint"](output, latest)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(campaign.pipeline, "train_classifier", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        campaign.run_experiment(**orchestration.arguments)
    monkeypatch.setattr(campaign.pipeline, "train_classifier", operation)
    result = campaign.run_experiment(**orchestration.arguments)
    kwargs = next(kwargs for name, kwargs in orchestration.calls if name == "classifier")
    assert kwargs["resume"] == Path(result["run"]) / "classifier/checkpoints/boundary-000000002.pt"


def test_partial_stage_without_checkpoint_is_preserved(orchestration, monkeypatch):
    def interrupt(**kwargs):
        kwargs["output"].mkdir()
        (kwargs["output"] / "user.txt").write_text("preserve")
        raise RuntimeError("interrupted before checkpoint")

    monkeypatch.setattr(campaign.pipeline, "train_classifier", interrupt)
    with pytest.raises(RuntimeError):
        campaign.run_experiment(**orchestration.arguments)
    with pytest.raises(ValueError, match="no safe durable checkpoint"):
        campaign.run_experiment(**orchestration.arguments)
    path = orchestration.arguments["work_root"] / "MSP/seed-0/classifier/user.txt"
    assert path.read_text() == "preserve"


def test_backup_failure_halts_without_claiming_completion(orchestration, monkeypatch):
    backup = campaign.backup.backup_run

    def fail_on_checkpoint(run, destination, metadata=None):
        if (run / "classifier").exists():
            raise OSError("backup unavailable")
        return backup(run, destination, metadata=metadata)

    monkeypatch.setattr(campaign.backup, "backup_run", fail_on_checkpoint)
    with pytest.raises(OSError, match="backup unavailable"):
        campaign.run_experiment(**orchestration.arguments)
    run = orchestration.arguments["work_root"] / "MSP/seed-0"
    assert not (run / "completed.json").exists()
    assert (run / "classifier/checkpoints/boundary-000000030.pt").is_file()
    assert not any(name == "evaluate" for name, _ in orchestration.calls)
    assert json.loads((run / "campaign.jsonl").read_text().splitlines()[-1])["event"] == "backup_failed"


def test_checkpoint_backup_first_then_interval(orchestration, monkeypatch):
    from oos_detect.reproduce.config import paper_config

    now = [0.0]
    monkeypatch.setattr(campaign.time, "monotonic", lambda: now[0])
    run = orchestration.arguments["work_root"] / "MSP/seed-0"
    output = run / "classifier"
    (output / "checkpoints").mkdir(parents=True)
    calls = []

    def verified(*args, **kwargs):
        calls.append(now[0])
        return {"files": {"checkpoint": {"size": 5}}, "generation": "test"}

    monkeypatch.setattr(campaign.backup, "backup_run", verified)
    snapshots = campaign._Backups(run, Path("unused"), {}, 600, paper_config("MSP"))
    snapshots.start_stage(output)
    snapshots.force("prepared-complete")
    for epoch, timestamp in ((1, 60.0), (2, 100.0), (3, 660.0)):
        now[0] = timestamp
        checkpoint = output / "checkpoints" / f"boundary-{epoch:09d}.pt"
        checkpoint.write_bytes(b"saved")
        snapshots.checkpoint(output, checkpoint)
    assert calls == [0.0, 60.0, 660.0]
    logs = [json.loads(line) for line in (run / "campaign.jsonl").read_text().splitlines()]
    progress = [row for row in logs if row["event"] == "checkpoint"]
    assert progress[-1]["seconds_per_epoch"] == 560.0
    assert progress[-1]["remaining_seconds"] == 27 * 560.0


def test_final_backup_failure_is_not_reported_as_success(orchestration, monkeypatch):
    backup = campaign.backup.backup_run

    def fail_final(run, destination, metadata=None):
        if (run / "completed.json").exists():
            raise OSError("final backup failed")
        return backup(run, destination, metadata=metadata)

    monkeypatch.setattr(campaign.backup, "backup_run", fail_final)
    with pytest.raises(OSError, match="final backup failed"):
        campaign.run_experiment(**orchestration.arguments)
    monkeypatch.setattr(campaign.backup, "backup_run", backup)
    orchestration.calls.clear()
    assert campaign.run_experiment(**orchestration.arguments)["complete"] is True
    assert orchestration.calls == []


def test_restore_changed_runtime_refuses_before_creating_run(orchestration, tmp_path):
    campaign.run_experiment(**orchestration.arguments)
    orchestration.identity["runtime"] = {"cuda": "different"}
    destination = tmp_path / "other-ssd"
    with pytest.raises(ValueError, match="metadata does not match"):
        campaign.run_experiment(**{**orchestration.arguments, "work_root": destination})
    assert not (destination / "MSP/seed-0").exists()


def test_cli_defaults_to_one_msp_seed_zero(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(campaign, "run_experiment", lambda **kwargs: calls.append(kwargs) or {
        "next_requested": {"experiment": "Perturb", "seed": 0}, "auto_advance": False})
    assert campaign.main(["--work-root", str(tmp_path / "ssd"),
                          "--backup-root", str(tmp_path / "drive"),
                          "--embeddings", "glove.txt"]) == 0
    assert len(calls) == 1 and calls[0]["experiment"] == "MSP" and calls[0]["seed"] == 0
    assert calls[0]["backup_seconds"] == 600 and calls[0]["checkpoint_seconds"] == 60
    assert json.loads(capsys.readouterr().out)["auto_advance"] is False


@pytest.mark.parametrize("arguments", [{"seed": 10}, {"seed": -1}, {"experiment": "all"},
                                       {"backup_seconds": 0}, {"checkpoint_seconds": float("nan")}])
def test_invalid_request_has_no_side_effects(orchestration, arguments):
    with pytest.raises(ValueError):
        campaign.run_experiment(**orchestration.arguments, **arguments)
    assert not orchestration.arguments["work_root"].exists()


def test_drive_work_root_is_rejected(orchestration):
    arguments = {**orchestration.arguments, "work_root": Path("/content/drive/MyDrive/work")}
    with pytest.raises(ValueError, match="local, not Drive"):
        campaign.run_experiment(**arguments)
    assert orchestration.calls == []
