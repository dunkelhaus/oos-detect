"""Artifact and CUDA integration checks. Run only in the approved CUDA runtime."""

from dataclasses import replace
import json

import pytest
import torch

from oos_detect.reproduce import pipeline
from oos_detect.reproduce.config import paper_config, smoke_config
from oos_detect.reproduce.data import SplitContract, prepare_data
from oos_detect.reproduce.runtime import require_cuda
from oos_detect.reproduce.state import derived_seed, file_hash, write_json


@pytest.fixture
def prepared(tmp_path):
    fixture = {}
    for partition, count in (("train", 4), ("val", 2), ("test", 2)):
        fixture[partition] = [[f"{label} request {partition} number {i}", label]
                              for label in ("book", "cancel") for i in range(count)]
        fixture[f"oos_{partition}"] = [[f"outside topic {partition} {i}", "oos"] for i in range(2)]
    full, plus = tmp_path / "full.json", tmp_path / "plus.json"
    write_json(full, fixture)
    write_json(plus, fixture)
    directory = tmp_path / "prepared"
    prepare_data(directory, full_path=full, plus_path=plus, seed=derived_seed(0, "prepare"),
                 contract=SplitContract(labels=2, train_per_label=4, val_per_label=2,
                                        test_per_label=2, full_ood_train=2, plus_ood_train=2,
                                        ood_val=2, ood_test=2), max_length=8, ind_count=4)
    return directory


def _load(path):
    return torch.load(path, map_location=require_cuda(), weights_only=False)


def _assert_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    elif isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            if key == "phase_log":  # Operational elapsed-time diagnostics are not training state.
                continue
            _assert_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            _assert_tree(first, second)
    else:
        assert actual == expected


def test_prepared_hashes_and_config_are_checked(prepared):
    config = smoke_config()
    artifacts = pipeline.load_prepared(prepared, config)
    assert artifacts.sha256 == artifacts.manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="length"):
        pipeline.load_prepared(prepared, replace(config, max_length=9))
    path = prepared / "split-train.json"
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash"):
        pipeline.load_prepared(prepared, config)


def test_generated_tokens_preserve_unk_and_full_cap(prepared):
    artifacts = pipeline.load_prepared(prepared, smoke_config())
    ids = pipeline.generated_token_ids("<unk> book", artifacts, 8)
    assert ids[:4] == (2, 1, artifacts.classifier.token_to_id["book"], 3)
    assert ids[4:] == (0, 0, 0, 0)
    ids = pipeline.generated_token_ids(" ".join(["book"] * 7), artifacts, 8)
    assert ids == (2,) + (artifacts.classifier.token_to_id["book"],) * 7
    for bad in ("<bos>", "<pad>", "book  book", "not-in-vocab", " ".join(["book"] * 8)):
        with pytest.raises(ValueError, match="space-joined"):
            pipeline.generated_token_ids(bad, artifacts, 8)


def test_phase_plan_has_all_repeats_and_excludes_dmix():
    config = replace(smoke_config("AEPOG"), critic_iterations=3,
                     dmix_reconstruction_iterations=2, dmix_critic_iterations=2)
    assert pipeline.generator_phases(config) == (
        "reconstruction", "auxiliary", "critic:0", "critic:1", "critic:2",
        "generator_wasserstein", "generator_entropy", "dmix_reconstruction:0",
        "dmix_reconstruction:1", "dmix_critic:0", "dmix_critic:1")
    for mode in ("pog", "pog-no-noise", "pog-no-soft-token"):
        assert all(not name.startswith("dmix") for name in pipeline.generator_phases(smoke_config(mode=mode)))


def test_loader_restores_permutation_cursor_and_rng():
    device = require_cuda()
    rng = torch.Generator(device=device).manual_seed(41)
    first = pipeline.BoundaryBatches(5, 2, rng)
    first.take()
    snapshot, rng_state = first.state_dict(), rng.get_state()
    expected = [first.take() for _ in range(5)]
    rng.set_state(rng_state)
    second = pipeline.BoundaryBatches(5, 2, rng)
    second.load_state_dict(snapshot)
    assert [second.take() for _ in range(5)] == expected
    with pytest.raises(ValueError, match="loader"):
        second.load_state_dict({**snapshot, "position": 99})


@pytest.mark.parametrize("mode,variant", [("pog", "POG"), ("aepog", "AEPOG"),
                                           ("pog-no-noise", "POG"), ("pog-no-soft-token", "POG")])
def test_phase_resume_matches_uninterrupted(prepared, tmp_path, mode, variant):
    require_cuda()
    config = smoke_config(variant, mode=mode)
    continuous = pipeline.train_generator(prepared=prepared, output=tmp_path / "continuous",
                                          config=config, seed=0)
    interrupted = pipeline.train_generator(prepared=prepared, output=tmp_path / "resumed",
                                           config=config, seed=0, stop_after_phases=4)
    partial = _load(interrupted["checkpoint"])
    assert partial["extra"]["phase_index"] == 4
    assert partial["extra"]["current_batches"]["ind"]
    assert partial["extra"]["loaders"]["ind"]["position"] == 2
    before = file_hash(interrupted["checkpoint"])
    resumed = pipeline.train_generator(prepared=prepared, output=tmp_path / "resumed",
                                       config=config, seed=0, resume=interrupted["checkpoint"])
    assert before == file_hash(interrupted["checkpoint"])
    actual, expected = _load(resumed["checkpoint"]), _load(continuous["checkpoint"])
    for section in ("models", "optimizers", "extra", "boundary", "temperature", "policy_baseline"):
        _assert_tree(actual[section], expected[section])
    for stream in actual["rng"]["streams"]:
        assert torch.equal(actual["rng"]["streams"][stream], expected["rng"]["streams"][stream])
    assert resumed["complete"] and actual["extra"]["generation"]["epoch"] == 1
    assert ("dmix_reconstruction" in resumed["update_counts"]) == (mode == "aepog")
    if mode == "pog-no-soft-token":
        assert actual["extra"]["trainer"]["baseline_updates"] == 4
    with pytest.raises(ValueError, match="latest"):
        pipeline.train_generator(prepared=prepared, output=tmp_path / "resumed", config=config,
                                 seed=0, resume=interrupted["checkpoint"])


def test_generator_epoch_boundary_resume_and_incompatible_config(prepared, tmp_path):
    config = replace(smoke_config(), generator_epochs=2)
    first = pipeline.train_generator(prepared=prepared, output=tmp_path / "run", config=config,
                                     seed=0, stop_after_phases=4 * len(pipeline.generator_phases(config)))
    assert first["completed_epochs"] == 1 and not first["complete"]
    assert _load(first["checkpoint"])["extra"]["current_batches"] == {}
    with pytest.raises(ValueError, match="Resume"):
        pipeline.train_generator(prepared=prepared, output=tmp_path / "run",
                                 config=replace(config, noise_std=0.2), seed=0, resume=first["checkpoint"])
    final = pipeline.train_generator(prepared=prepared, output=tmp_path / "run", config=config,
                                     seed=0, resume=first["checkpoint"])
    assert final["complete"] and final["completed_epochs"] == 2


def test_bounded_generator_logs_resume_and_callback(prepared, tmp_path):
    require_cuda()
    config = smoke_config()
    output = tmp_path / "bounded"
    snapshots = []

    def snapshot(run, checkpoint):
        assert run == output and checkpoint.is_file()
        logs = [json.loads(line) for line in (run / "phases.jsonl").read_text().splitlines()]
        assert logs[-1]["checkpoint"] == checkpoint.name
        assert len(list((run / "checkpoints").glob("boundary-*.pt"))) == 1
        snapshots.append(checkpoint)

    partial = pipeline.train_generator(
        prepared=prepared, output=output, config=config, seed=0, stop_after_phases=4,
        checkpoint_interval_seconds=3600, retain_checkpoints=True, on_checkpoint=snapshot)
    assert len(snapshots) == 1
    log = output / "phases.jsonl"
    rows = log.read_text().splitlines()
    assert len(rows) == 4
    # Simulate a saved checkpoint followed by a missing log and speculative logs.
    log.write_text("\n".join(rows[:3]) + "\n" + json.dumps(
        {"phase": "critic", "boundary_serial": 5, "loss": 123}) + "\n")
    final = pipeline.train_generator(
        prepared=prepared, output=output, config=config, seed=0, resume=partial["checkpoint"],
        checkpoint_interval_seconds=3600, retain_checkpoints=True, on_checkpoint=snapshot)
    assert final["complete"]
    logs = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["boundary_serial"] for row in logs] == list(range(1, final["phase_boundaries"] + 1))
    assert len(snapshots) == 2 and (output / "summary.json").is_file()
    continuous = pipeline.train_generator(prepared=prepared, output=tmp_path / "continuous-bounded",
                                          config=config, seed=0)
    actual, expected = _load(final["checkpoint"]), _load(continuous["checkpoint"])
    for section in ("models", "optimizers", "extra", "boundary", "temperature", "policy_baseline"):
        _assert_tree(actual[section], expected[section])


def test_classifier_retention_preserves_evaluation(prepared, tmp_path):
    require_cuda()
    config = replace(smoke_config("MSP"), classifier_epochs=3)
    output = tmp_path / "bounded-classifier"
    calls = []

    def snapshot(run, checkpoint):
        references = json.loads((run / "checkpoint-policy.json").read_text())["references"]
        assert checkpoint.name == references["latest"]
        assert len(list((run / "checkpoints").glob("boundary-*.pt"))) <= 3
        assert all((run / "checkpoints" / name).is_file() for name in references.values())
        calls.append(checkpoint)

    result = pipeline.train_classifier(prepared=prepared, output=output, config=config, seed=0,
                                       checkpoint_interval_seconds=60, retain_checkpoints=True,
                                       on_checkpoint=snapshot)
    assert len(calls) == 3
    assert (output / result["selected_checkpoint"]).is_file()
    assert (output / result["final_checkpoint"]).is_file()
    pipeline.evaluate(prepared=prepared, classifier_run=output, output=tmp_path / "bounded-evaluation",
                      config=config, seed=0)


def test_classifier_epoch_resume_matches_uninterrupted(prepared, tmp_path):
    config = replace(smoke_config("Perturb"), classifier_epochs=2)
    continuous = pipeline.train_classifier(prepared=prepared, output=tmp_path / "continuous", config=config, seed=0)
    first = pipeline.train_classifier(prepared=prepared, output=tmp_path / "resumed", config=config,
                                      seed=0, stop_after_epochs=1)
    assert not first["complete"]
    final = pipeline.train_classifier(prepared=prepared, output=tmp_path / "resumed", config=config,
                                      seed=0, resume=first["checkpoint"])
    actual, expected = _load(final["checkpoint"]), _load(continuous["checkpoint"])
    for section in ("models", "optimizers", "extra", "boundary"):
        _assert_tree(actual[section], expected[section])
    assert (tmp_path / "resumed" / final["selected_checkpoint"]).is_file()
    assert (tmp_path / "resumed" / final["final_checkpoint"]).is_file()


@pytest.mark.parametrize("variant,mode", [("MSP", "pog"), ("Perturb", "pog"), ("Mix", "pog"),
                                          ("POG", "pog"), ("AEPOG", "aepog"),
                                          ("POG", "pog-no-noise"), ("POG", "pog-no-soft-token")])
def test_artifact_driven_flow(prepared, tmp_path, variant, mode):
    require_cuda()
    config = smoke_config(variant, mode=mode)
    generator_run = generated = None
    if config.uses_generator:
        generator_run, generated = tmp_path / "generator", tmp_path / "generated"
        pipeline.train_generator(prepared=prepared, output=generator_run, config=config, seed=0)
        exported = pipeline.generate(prepared=prepared, output=generated, generator_run=generator_run,
                                     config=config, seed=0)
        assert exported["manifest"]["count"] == 8
        assert exported["manifest"]["profile"] == "tiny-non-paper-generation"
    trained = pipeline.train_classifier(prepared=prepared, output=tmp_path / "classifier", config=config,
                                        seed=0, generated=generated, generator_run=generator_run)
    assert trained["complete"]
    report = pipeline.evaluate(prepared=prepared, classifier_run=tmp_path / "classifier",
                               output=tmp_path / "evaluation", config=config, seed=0, calibrate=True)
    assert report["split"] == "test"
    assert report["class_balance"] == {"IND": 4, "OOD": 2}
    assert report["deployment_threshold"]["calibration"]["split"] == "val"
    assert report["deviations"]
    aggregated = pipeline.aggregate(reports=[tmp_path / "evaluation/summary.json"],
                                     output=tmp_path / "aggregate.json", expected_seeds=(0, 1))
    assert aggregated["status"] == "incomplete" and aggregated["missing_seeds"] == [1]
    assert not aggregated["numerical_parity_gate"]
    manifest = json.loads((tmp_path / "classifier/manifest.json").read_text())
    assert (manifest["config"]["inputs"]["entropy"] is None) == (variant == "MSP")
    assert json.loads((tmp_path / "classifier/embeddings.json").read_text())["classifier"]["profile"] == "random-only-smoke-non-paper"
    before = file_hash(tmp_path / "classifier/manifest.json")
    with pytest.raises(FileExistsError):
        pipeline.train_classifier(prepared=prepared, output=tmp_path / "classifier", config=config,
                                  seed=0, generated=generated, generator_run=generator_run)
    assert before == file_hash(tmp_path / "classifier/manifest.json")
    if config.uses_generator:
        corpus_path = generated / "corpus.jsonl"
        corpus_path.write_text(corpus_path.read_text() + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="corpus hash"):
            pipeline.train_classifier(prepared=prepared, output=tmp_path / "bad-corpus", config=config,
                                      seed=0, generated=generated, generator_run=generator_run)
        assert not (tmp_path / "bad-corpus").exists()


def test_missing_prerequisites_do_not_create_outputs(prepared, tmp_path):
    with pytest.raises(FileNotFoundError, match="glove"):
        pipeline.train_generator(prepared=prepared, output=tmp_path / "paper", config=paper_config(), seed=0)
    with pytest.raises(ValueError, match="require --generated"):
        pipeline.train_classifier(prepared=prepared, output=tmp_path / "missing", config=smoke_config(), seed=0)
    assert not (tmp_path / "paper").exists() and not (tmp_path / "missing").exists()


def test_unfinished_generator_cannot_export(prepared, tmp_path):
    config = smoke_config()
    pipeline.train_generator(prepared=prepared, output=tmp_path / "generator", config=config,
                             seed=0, stop_after_phases=1)
    with pytest.raises(FileNotFoundError):
        pipeline.generate(prepared=prepared, generator_run=tmp_path / "generator", output=tmp_path / "generated",
                          config=config, seed=0)
    assert not (tmp_path / "generated").exists()


def test_evaluation_checks_final_selection_before_test_inference(prepared, tmp_path, monkeypatch):
    from oos_detect.reproduce import classifier

    config = smoke_config("MSP")
    run = tmp_path / "classifier"
    pipeline.train_classifier(prepared=prepared, output=run, config=config, seed=0)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["best_validation_accuracy"] = 1.1
    path.write_text(json.dumps(summary), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Test prediction must not run before selection validation")

    monkeypatch.setattr(classifier, "predict_confidence", forbidden)
    with pytest.raises(ValueError, match="final validation-selected"):
        pipeline.evaluate(prepared=prepared, classifier_run=run, output=tmp_path / "evaluation", config=config, seed=0)
    assert not (tmp_path / "evaluation").exists()


def test_smoke_command_initializes_cuda_in_a_fresh_process(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "oos_detect.reproduce", "smoke",
         "--output", str(tmp_path / "smoke"), "--device", "cuda:0"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert len(report["cases"]) == 7
    assert report["peak_allocated_bytes"] > 0
    assert report["profile"] == "tiny-smoke-non-paper"
