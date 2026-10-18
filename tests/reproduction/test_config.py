"""Offline profile contracts; these tests do not construct models."""

import json
from pathlib import Path

import pytest

from oos_detect.reproduce.config import (
    MODES, VARIANTS, load_config, paper_config, resolve_config, smoke_config,
    validate_prerequisites,
)


@pytest.mark.parametrize("variant", VARIANTS)
def test_paper_profiles(variant):
    config = paper_config(variant)
    assert config.generator_epochs == 80
    assert config.classifier_epochs == 30
    assert config.seeds == tuple(range(10))
    assert config.generated_count == 15000
    assert not config.random_only
    assert config.classifier_settings(9)["paper"] is True
    assert config.pog_settings()["max_steps"] == 63
    assert config.for_seed(9)["seed"] == 9
    assert resolve_config(config.to_dict()) == config


@pytest.mark.parametrize("mode", MODES)
def test_generator_modes_and_smoke(mode):
    variant = "AEPOG" if mode == "aepog" else "POG"
    paper = paper_config(variant, mode=mode)
    smoke = smoke_config(variant, mode=mode)
    assert paper.uses_dmix == (mode == "aepog")
    assert smoke.profile == "tiny-smoke-non-paper"
    assert smoke.generator_epochs == smoke.classifier_epochs == 1
    assert smoke.random_only and smoke.seeds == (0,)
    assert smoke.embedding_dim == 100  # Matches the existing embedding loader.
    assert smoke.model_kwargs()["code_dim"] == 8
    assert smoke.pog_settings()["mode"] == mode
    assert not smoke.classifier_settings(0)["paper"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_baseline_has_an_explicit_smoke_profile(variant):
    assert smoke_config(variant).variant == variant


@pytest.mark.parametrize("changes", [
    {"generator_epochs": 1}, {"classifier_epochs": 1}, {"seeds": [0]},
    {"generated_count": 8}, {"random_only": True}, {"embedding_dim": 8},
    {"batch_size": 2}, {"vocabulary_profile": "train-only"},
    {"alpha": 0}, {"perturb_probability": 0.1}, {"noise_std": 0},
    {"train_limit": 8}, {"critic_iterations": 1},
])
def test_paper_never_silently_changes_experiment(changes):
    with pytest.raises(ValueError, match="Paper profile requires"):
        resolve_config(changes)


@pytest.mark.parametrize("changes", [
    {"variant": "other"}, {"mode": "argmax"}, {"device": "cpu"},
    {"batch_size": True}, {"seeds": [False]}, {"seeds": [0, 0]},
    {"seeds": []}, {"seeds": [-1]}, {"seeds": "0"}, {"ae_lr": float("nan")},
    {"gp_weight": float("inf")}, {"classifier_lr": True}, {"gan_betas": [0, 1]},
    {"kernel_sizes": []}, {"mode": "aepog"},
    {"variant": "Mix", "mode": "pog-no-noise"}, {"unknown": 1},
])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        resolve_config(changes)


def test_explicit_seed_selection():
    with pytest.raises(ValueError, match="declared campaign seed"):
        paper_config().for_seed(10)
    with pytest.raises(ValueError):
        paper_config().classifier_settings(True)


def test_strict_json(tmp_path):
    path = tmp_path / "config.json"
    for content in ('{"batch_size":32,"batch_size":2}', '{"ae_lr":NaN}'):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text(json.dumps(smoke_config().to_dict()), encoding="utf-8")
    assert load_config(path) == smoke_config()


def test_supplied_json_profiles():
    root = Path(__file__).resolve().parents[2] / "configs/reproduction"
    assert load_config(root / "paper-pog.json") == paper_config()
    assert load_config(root / "tiny-smoke-pog.json") == smoke_config()


@pytest.mark.parametrize("stage", ["train-generator", "train-classifier"])
def test_paper_missing_embeddings_fail_before_models(stage, tmp_path):
    with pytest.raises(FileNotFoundError, match="no download or random-vector fallback"):
        validate_prerequisites(paper_config(), stage, embedding_path=tmp_path / "missing")
    # Text preparation does not require tensor initialization or pretrained files.
    validate_prerequisites(paper_config(), "prepare")


def test_missing_generated_text_and_forbidden_sources(tmp_path):
    embedding = tmp_path / "glove.6B.100d.txt"
    embedding.write_text("placeholder; actual vectors are validated by load_embeddings", encoding="utf-8")
    with pytest.raises(ValueError, match="separate pseudo-OOD text corpus"):
        validate_prerequisites(paper_config(), "train-classifier", embedding_path=embedding)
    for variant in ("MSP", "Perturb", "Mix"):
        validate_prerequisites(smoke_config(variant), "train-classifier")
        with pytest.raises(ValueError, match="forbids a generated"):
            validate_prerequisites(smoke_config(variant), "train-classifier", generated_manifest={})
        with pytest.raises(ValueError, match="no generator stage"):
            validate_prerequisites(smoke_config(variant), "train-generator")
    with pytest.raises(ValueError, match="forbids a pretrained"):
        validate_prerequisites(smoke_config(), "train-generator", embedding_path=embedding)


def corpus_inputs(config):
    """Use export_generated_corpus/checkpoint_generation_metadata field names."""
    hashes = {"data_sha256": "a" * 64, "vocabulary_sha256": "b" * 64,
              "checkpoint_sha256": "c" * 64}
    manifest = {"format_version": 1, **hashes,
                "generator_vocabulary_sha256": "d" * 64,
                "selection": "final-configured-epoch",
                "variant": config.variant, "mode": config.mode,
                "profile": "paper-generation" if config.paper else "tiny-non-paper-generation",
                "epoch": config.generator_epochs, "epochs": config.generator_epochs,
                "count": config.generated_count}
    records = [{"id": f"generated:{i}", "text": "duplicate text is allowed"}
               for i in range(config.generated_count)]
    return {**hashes, "generated_manifest": manifest, "generated_records": records}


@pytest.mark.parametrize("mode", MODES)
def test_corpus_provenance_for_every_mode(mode):
    config = smoke_config("AEPOG" if mode == "aepog" else "POG", mode=mode)
    inputs = corpus_inputs(config)
    validate_prerequisites(config, "train-classifier", **inputs)
    for key, value in (("mode", "wrong"), ("profile", "paper"), ("count", 15000),
                       ("epoch", 80), ("epochs", 80), ("variant", "other"),
                       ("data_sha256", "d" * 64), ("vocabulary_sha256", "d" * 64),
                       ("checkpoint_sha256", "d" * 64)):
        changed = {**inputs, "generated_manifest": {**inputs["generated_manifest"], key: value}}
        with pytest.raises(ValueError):
            validate_prerequisites(config, "train-classifier", **changed)
    for records in (inputs["generated_records"][:-1],
                    [{"id": str(i), "text": "x", "target": 0} for i in range(config.generated_count)],
                    [{"id": "same", "text": "x"}] * config.generated_count):
        with pytest.raises(ValueError):
            validate_prerequisites(config, "train-classifier", **{**inputs, "generated_records": records})


def test_paper_corpus_count_checked_against_actual_records(tmp_path):
    config = paper_config()
    path = tmp_path / "glove.txt"
    path.touch()
    inputs = corpus_inputs(config)
    validate_prerequisites(config, "train-classifier", embedding_path=path, **inputs)
    inputs["generated_records"] = inputs["generated_records"][:8]
    with pytest.raises(ValueError, match="configured number"):
        validate_prerequisites(config, "train-classifier", embedding_path=path, **inputs)


def test_boolean_epoch_is_not_a_completed_epoch():
    config = smoke_config()
    inputs = corpus_inputs(config)
    inputs["generated_manifest"]["epoch"] = True
    with pytest.raises(ValueError, match="final-epoch"):
        validate_prerequisites(config, "train-classifier", **inputs)


@pytest.mark.parametrize("paper", [True, False])
@pytest.mark.parametrize("mode", MODES)
def test_export_stage_profile_required_for_every_mode(paper, mode, tmp_path):
    variant = "AEPOG" if mode == "aepog" else "POG"
    config = (paper_config if paper else smoke_config)(variant, mode=mode)
    inputs = corpus_inputs(config)
    embedding = tmp_path / "glove.txt"
    embedding.touch()  # Vector content checks belong to data.load_embeddings.
    prerequisites = {"embedding_path": embedding} if paper else {}
    expected = "paper-generation" if paper else "tiny-non-paper-generation"
    assert inputs["generated_manifest"]["profile"] == expected
    validate_prerequisites(config, "train-classifier", **prerequisites, **inputs)
    # Neither a campaign label nor another stage's artifact is a generated profile.
    for wrong in ("paper", "tiny-smoke-non-paper", "paper-text-preparation",
                  "non-paper", "paper-generation", "tiny-non-paper-generation", None):
        if wrong == expected:
            continue
        manifest = {**inputs["generated_manifest"], "profile": wrong}
        with pytest.raises(ValueError, match="profile"):
            validate_prerequisites(config, "train-classifier", **prerequisites,
                                   **{**inputs, "generated_manifest": manifest})
    # An export with the right stage profile still cannot cross ablation modes.
    for wrong_mode in MODES:
        if wrong_mode == mode:
            continue
        manifest = {**inputs["generated_manifest"], "mode": wrong_mode}
        with pytest.raises(ValueError, match="mode"):
            validate_prerequisites(config, "train-classifier", **prerequisites,
                                   **{**inputs, "generated_manifest": manifest})
