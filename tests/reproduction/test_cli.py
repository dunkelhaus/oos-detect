"""Command wiring; model execution is covered by the CUDA acceptance suite."""

from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from oos_detect.reproduce.cli import main


def test_prepare_command_arguments(monkeypatch, tmp_path, capsys):
    from oos_detect.reproduce import data

    calls = []

    def prepare(output, **kwargs):
        calls.append((output, kwargs))
        return SimpleNamespace(manifest={"manifest_sha256": "example"})

    monkeypatch.setattr(data, "prepare_data", prepare)
    output = tmp_path / "prepared"
    full, plus = tmp_path / "data_full.json", tmp_path / "data_oos_plus.json"
    assert main(["prepare", "--output", str(output), "--full", str(full),
                 "--plus", str(plus), "--seed", "7", "--profile", "train-only",
                 "--max-length", "12", "--dmix-ind-count", "10"]) == 0
    assert calls == [(output, {"seed": 7, "profile": "train-only", "max_length": 12,
                               "ind_count": 10, "full_path": full, "plus_path": plus})]
    assert json.loads(capsys.readouterr().out)["manifest"]["manifest_sha256"] == "example"


def test_prepare_defaults_remain_paper_text_profile(monkeypatch, tmp_path):
    from oos_detect.reproduce import data

    calls = []

    def prepare(output, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(manifest={})

    monkeypatch.setattr(data, "prepare_data", prepare)
    main(["prepare", "--output", str(tmp_path / "new")])
    assert calls == [{"seed": 0, "max_length": 64, "profile": "paper", "ind_count": 10000}]


@pytest.mark.parametrize("command,extra,operation", [
    ("train-generator", [], "train_generator"),
    ("generate", ["--generator-run", "generator"], "generate"),
    ("train-classifier", ["--generator-run", "generator", "--generated", "corpus"], "train_classifier"),
    ("evaluate", ["--classifier-run", "classifier", "--calibrate"], "evaluate"),
])
def test_artifact_stage_wiring(monkeypatch, capsys, command, extra, operation):
    from oos_detect.reproduce import pipeline

    calls = []
    monkeypatch.setattr(pipeline, operation, lambda **kwargs: calls.append(kwargs) or {"stage": command})
    assert main([command, "--prepared", "prepared", "--output", "output", "--seed", "0",
                 "--profile", "tiny-smoke-non-paper", *extra]) == 0
    assert len(calls) == 1
    assert calls[0]["prepared"] == Path("prepared")
    assert calls[0]["config"].random_only and calls[0]["config"].generator_epochs == 1
    if command == "evaluate":
        assert calls[0]["calibrate"] is True
    assert json.loads(capsys.readouterr().out) == {"stage": command}


def test_paper_defaults_never_launch_campaign(monkeypatch):
    from oos_detect.reproduce import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "train_generator", lambda **kwargs: calls.append(kwargs) or {})
    main(["train-generator", "--prepared", "prepared", "--output", "output", "--seed", "3",
          "--embeddings", "glove.txt", "--resume", "boundary.pt", "--generator-epochs", "80",
          "--classifier-epochs", "30", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    assert len(calls) == 1 and calls[0]["seed"] == 3
    assert calls[0]["config"].seeds == tuple(range(10))
    assert calls[0]["config"].generator_epochs == 80
    assert calls[0]["config"].classifier_epochs == 30
    assert calls[0]["embedding_path"] == Path("glove.txt")
    assert calls[0]["resume"] == Path("boundary.pt")


def test_json_config_cannot_have_hidden_cli_overrides(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["train-generator", "--prepared", "prepared", "--output", "output", "--seed", "0",
              "--config", str(config), "--device", "cuda:1"])
    assert error.value.code == 2


def test_smoke_and_aggregate_are_explicit(monkeypatch):
    from oos_detect.reproduce import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "smoke", lambda **kwargs: calls.append(("smoke", kwargs)) or {})
    monkeypatch.setattr(pipeline, "aggregate", lambda **kwargs: calls.append(("aggregate", kwargs)) or {})
    main(["aggregate", "--reports", "a.json", "b.json", "--output", "summary.json",
          "--expected-seeds", "0", "1", "2"])
    assert calls == [("aggregate", {"reports": [Path("a.json"), Path("b.json")],
                                     "output": Path("summary.json"), "expected_seeds": (0, 1, 2)})]
    main(["smoke", "--output", "smoke", "--device", "cuda:1"])
    assert calls[-1] == ("smoke", {"output": Path("smoke"), "device": "cuda:1"})


def test_help_does_not_call_any_pipeline_stage(monkeypatch, capsys):
    from oos_detect.reproduce import pipeline

    def unexpected(**kwargs):
        raise AssertionError("Help must not execute a stage")

    for operation in ("train_generator", "generate", "train_classifier", "evaluate", "aggregate", "smoke"):
        monkeypatch.setattr(pipeline, operation, unexpected)
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "train-generator" in capsys.readouterr().out
