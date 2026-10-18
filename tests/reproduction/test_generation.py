"""CUDA generation checks. Execution is reserved for the parent Colab session."""

import json

import pytest
import torch

from pseudo_ood_generation.builders import build_model
from oos_detect.reproduce.data import SPECIAL_TOKENS, Vocabulary, content_hash
from oos_detect.reproduce.generation import (
    GenerationConfig, _text_from_tokens, checkpoint_generation_metadata,
    export_generated_corpus, load_generated_corpus, load_generation_checkpoint,
)
from oos_detect.reproduce.runtime import require_cuda
from oos_detect.reproduce.state import Boundary, file_hash, fingerprint, save_checkpoint


@pytest.fixture
def generation_case(tmp_path):
    device = require_cuda()
    generator_vocab = Vocabulary(SPECIAL_TOKENS + ("alpha", "beta"))
    classifier_vocab = Vocabulary(SPECIAL_TOKENS + ("alpha", "beta", "gamma"))
    prepared = {"profile": "non-paper", "vocabulary_hashes": {
        "generator": generator_vocab.sha256, "classifier": classifier_vocab.sha256},
        "tokenizer": {"max_length": 8}}
    prepared["manifest_sha256"] = content_hash(prepared)
    run = {"config": {"seed": 4, "mode": "pog", "epochs": 2},
           "data_hash": prepared["manifest_sha256"], "embedding_hash": "fixture-only"}
    run["compatibility_hash"] = fingerprint(run)
    model = build_model(6, 7, list(range(6)), embedding_dim=4, code_dim=3,
                        noise_dim=3, gan_hidden_dim=5, gan_layers=2,
                        num_classes=2, out_channels=2, mlp_hidden_size=5, device=str(device))
    # Sample UNK/alpha/beta, never EOS, so the fixture cannot fail through chance empties.
    with torch.no_grad():
        model.autoencoder.decoder.projection.weight.zero_()
        model.autoencoder.decoder.projection.bias.zero_()
        model.autoencoder.decoder.projection.bias[3] = -1000
    metadata = checkpoint_generation_metadata(
        mode="pog", completed_epochs=2, configured_epochs=2,
        data_sha256=prepared["manifest_sha256"], generator_vocabulary=generator_vocab,
        classifier_vocabulary=classifier_vocab)
    return {"model": model, "run_manifest": run, "prepared_manifest": prepared,
            "generator_vocabulary": generator_vocab, "classifier_vocabulary": classifier_vocab,
            "configured_epochs": 2}, metadata


def save_case(tmp_path, case, metadata, *, model_key="pog", name="generator.pt"):
    path = tmp_path / name
    model = case["model"]
    save_checkpoint(path, models=model.phase_modules if model_key is None else {model_key: model},
                    optimizers={}, generators={}, boundary=Boundary(1, 1, "complete"),
                    compatibility_hash=case["run_manifest"]["compatibility_hash"],
                    temperature=0.1, policy_baseline=0,
                    extra={"generation": metadata})
    return path


def export(tmp_path, case, checkpoint, name="export", seed=17, **overrides):
    settings = dict(seed=seed, paper=False, count=9, max_length=8, batch_size=3)
    settings.update(overrides)
    return export_generated_corpus(tmp_path / name, checkpoint=checkpoint,
                                   config=GenerationConfig(**settings), **case)


def reader_arguments(corpus):
    return {key: corpus.manifest[key] for key in (
        "checkpoint_sha256", "data_sha256", "vocabulary_sha256",
        "generator_vocabulary_sha256", "variant")} | {"configured_epochs": 2, "paper": False}


def test_paper_count_and_explicit_tiny():
    assert GenerationConfig(seed=0).count == 15000
    with pytest.raises(ValueError, match="15000"):
        GenerationConfig(seed=0, count=2)
    assert GenerationConfig(seed=0, count=2, paper=False).count == 2
    with pytest.raises(ValueError, match="CUDA"):
        GenerationConfig(seed=0, device="cpu")


@pytest.mark.parametrize("model_key", ["pog", "model", None])
def test_export_uses_existing_checkpoint_layout_and_independent_rng(tmp_path, generation_case, model_key):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata, model_key=model_key)
    original_hash = file_hash(checkpoint)
    model = case["model"]
    model.train()
    model.auxiliary.eval()
    flags = [m.training for m in model.modules()]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    first = export_generated_corpus(
        tmp_path / "first", checkpoint=checkpoint, model_key=model_key,
        config=GenerationConfig(seed=17, paper=False, count=9, max_length=8, batch_size=3), **case)
    second = export_generated_corpus(
        tmp_path / "second", checkpoint=checkpoint, model_key=model_key,
        config=GenerationConfig(seed=17, paper=False, count=9, max_length=8, batch_size=3), **case)
    assert first.records == second.records
    assert (tmp_path / "first/corpus.jsonl").read_bytes() == (tmp_path / "second/corpus.jsonl").read_bytes()
    assert first.manifest == second.manifest
    assert first.manifest["profile"] == "tiny-non-paper-generation"
    assert first.manifest["diagnostics"]["truncated_records"] == 9
    assert first.manifest["diagnostics"]["attempts"] == 9
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert [m.training for m in model.modules()] == flags
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    assert file_hash(checkpoint) == original_hash
    assert all(set(json.loads(line)) == {"id", "text"}
               for line in (tmp_path / "first/corpus.jsonl").read_text().splitlines())


def test_seed_changes_sampling(tmp_path, generation_case):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    first = export(tmp_path, case, checkpoint, "first", seed=17)
    second = export(tmp_path, case, checkpoint, "second", seed=31)
    assert first.records != second.records


def test_duplicate_and_unk_diagnostics(tmp_path, generation_case):
    case, metadata = generation_case
    with torch.no_grad():
        case["model"].autoencoder.decoder.projection.bias.fill_(-1000)
        case["model"].autoencoder.decoder.projection.bias[1] = 0
    checkpoint = save_case(tmp_path, case, metadata)
    corpus = export(tmp_path, case, checkpoint)
    assert corpus.manifest["diagnostics"] == {
        "attempts": 9, "empty_outputs": 0, "duplicate_records": 8,
        "unk_tokens": 63, "records_with_unk": 9, "truncated_records": 9}
    assert len({row.id for row in corpus.records}) == 9


def test_eos_stops_export_text():
    vocab = Vocabulary(SPECIAL_TOKENS + ("word",))
    assert _text_from_tokens([4, 3, 0, 0], vocab) == ("word", 0, False)
    assert _text_from_tokens([3, 0], vocab) == ("", 0, False)
    assert _text_from_tokens([1, 4], vocab) == ("<unk> word", 1, True)
    with pytest.raises(ValueError, match="PAD/BOS"):
        _text_from_tokens([2, 4], vocab)


def test_empty_retry_failure_does_not_publish_success(tmp_path, generation_case):
    case, metadata = generation_case
    with torch.no_grad():
        case["model"].autoencoder.decoder.projection.bias.fill_(-1000)
        case["model"].autoencoder.decoder.projection.bias[3] = 0
    checkpoint = save_case(tmp_path, case, metadata)
    case["model"].train()
    with pytest.raises(RuntimeError, match="retry limit.*2 retries"):
        export(tmp_path, case, checkpoint, max_empty_retries=2)
    assert not (tmp_path / "export/manifest.json").exists()
    assert case["model"].training


def test_empty_slots_retry_with_fresh_noise_and_sample_policy(tmp_path, generation_case, monkeypatch):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    sizes, noises = [], []
    generator = case["model"].generator
    original_sample = generator.sample_latent

    def sample(size, *, rng):
        noise = original_sample(size, rng=rng)
        noises.append(noise.clone())
        return noise

    def rollout(codes, *, max_steps, temperature, mode, generator):
        assert mode == "sample" and temperature == 1.0 and max_steps == 7
        assert generator.device.type == "cuda"
        sizes.append(codes.shape[0])
        rows = [[4, 3, 0, 0, 0, 0, 0] for _ in range(codes.shape[0])]
        if len(sizes) == 1:
            rows[0] = [3, 0, 0, 0, 0, 0, 0]
        return {"tokens": torch.tensor(rows, device=codes.device)}

    monkeypatch.setattr(generator, "sample_latent", sample)
    monkeypatch.setattr(case["model"].autoencoder, "rollout", rollout)
    corpus = export(tmp_path, case, checkpoint, count=3, max_empty_retries=1)
    assert sizes == [3, 1]
    assert not torch.equal(noises[0][:1], noises[1])
    assert corpus.manifest["diagnostics"]["empty_outputs"] == 1
    assert corpus.manifest["diagnostics"]["attempts"] == 4
    assert corpus.manifest["diagnostics"]["truncated_records"] == 0
    assert all(row.text == "alpha" for row in corpus.records)


@pytest.mark.parametrize("key,value", [("epoch", 1), ("epochs", 3),
                                         ("generator_vocabulary_sha256", "wrong"),
                                         ("vocabulary_sha256", "wrong"),
                                         ("data_sha256", "wrong")])
def test_checkpoint_provenance_rejected(tmp_path, generation_case, key, value):
    case, metadata = generation_case
    metadata[key] = value
    checkpoint = save_case(tmp_path, case, metadata)
    with pytest.raises(ValueError, match="final configured epoch"):
        load_generation_checkpoint(checkpoint, **case)


def test_checkpoint_compatibility_rejected(tmp_path, generation_case):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    case["run_manifest"]["compatibility_hash"] = "wrong"
    with pytest.raises(ValueError, match="compatibility"):
        load_generation_checkpoint(checkpoint, **case)


def test_reader_prerequisites_and_output_collision(tmp_path, generation_case):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    corpus = export(tmp_path, case, checkpoint)
    arguments = reader_arguments(corpus)
    assert load_generated_corpus(tmp_path / "export", **arguments).records == corpus.records
    for field in ("checkpoint_sha256", "data_sha256", "vocabulary_sha256", "generator_vocabulary_sha256"):
        with pytest.raises(ValueError, match="provenance"):
            load_generated_corpus(tmp_path / "export", **(arguments | {field: "wrong"}))
    original = (tmp_path / "export/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        export(tmp_path, case, checkpoint)
    assert (tmp_path / "export/manifest.json").read_bytes() == original
    with pytest.raises(FileNotFoundError, match="generate the corpus"):
        load_generated_corpus(tmp_path / "missing", **arguments)


def test_reader_rejects_targets_even_with_updated_hashes(tmp_path, generation_case):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    corpus = export(tmp_path, case, checkpoint)
    path = tmp_path / "export/corpus.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["target"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = dict(corpus.manifest)
    manifest["corpus_sha256"] = file_hash(path)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = fingerprint(manifest)
    (tmp_path / "export/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target-free"):
        load_generated_corpus(tmp_path / "export", **reader_arguments(corpus))


@pytest.mark.parametrize("filename", ["corpus.jsonl", "manifest.json"])
def test_reader_rejects_changed_artifact_bytes(tmp_path, generation_case, filename):
    case, metadata = generation_case
    checkpoint = save_case(tmp_path, case, metadata)
    corpus = export(tmp_path, case, checkpoint)
    path = tmp_path / "export" / filename
    if filename == "manifest.json":
        manifest = json.loads(path.read_text())
        manifest["count"] = 10
        path.write_text(json.dumps(manifest))
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_generated_corpus(tmp_path / "export", **reader_arguments(corpus))


@pytest.mark.parametrize("mode", ["pog-no-noise", "pog-no-soft-token", "aepog"])
def test_variant_provenance(tmp_path, generation_case, mode):
    case, metadata = generation_case
    metadata = checkpoint_generation_metadata(
        mode=mode, completed_epochs=2, configured_epochs=2,
        data_sha256=case["prepared_manifest"]["manifest_sha256"],
        generator_vocabulary=case["generator_vocabulary"],
        classifier_vocabulary=case["classifier_vocabulary"])
    checkpoint = save_case(tmp_path, case, metadata)
    corpus = export(tmp_path, case, checkpoint)
    assert corpus.manifest["mode"] == mode
    assert corpus.manifest["variant"] == metadata["variant"]
