"""Offline CLINC contracts; only embedding tests require CUDA."""

from dataclasses import asdict
import json

import pytest

from oos_detect.reproduce.data import (
    BOS, EOS, PAD, UNK, ORIGINAL_DATA, SplitContract,
    build_dmix, build_vocabularies, build_vocabulary, canonical_json,
    content_hash, encode_text, file_hash, load_corpus, load_embeddings,
    prepare_data, tokenize,
)


@pytest.fixture
def originals(tmp_path):
    """Create original-format tiny sources with distinct held-out text."""
    contract = SplitContract(2, 3, 1, 1, 1, 2, 1, 1)
    full = {"train": [[f"train {label} {index}", label]
                      for label in ("zeta", "alpha") for index in range(3)],
            "val": [[f"validationonly {label}", label] for label in ("alpha", "zeta")],
            "test": [[f"testonly {label}", label] for label in ("alpha", "zeta")],
            "oos_train": [["fulloutlier", "oos"]],
            "oos_val": [["oodvalidation", "oos"]],
            "oos_test": [["oodtest", "oos"]]}
    plus = {**full, "oos_train": [["plusoutlier one", "oos"], ["plusoutlier two", "oos"]]}
    paths = (tmp_path / "data_full.json", tmp_path / "data_oos_plus.json")
    for path, value in zip(paths, (full, plus)):
        path.write_bytes(canonical_json(value))
    return paths, contract


def tiny_corpus(originals):
    paths, contract = originals
    return load_corpus(*paths, contract=contract)


def test_original_validation_and_sorted_labels(originals):
    corpus = tiny_corpus(originals)
    assert corpus.label_map == {"alpha": 0, "zeta": 1}
    assert corpus.full["train"][0].target == 1
    assert corpus.full["oos_test"][0].target is None
    assert corpus.full["oos_test"][0].is_ood
    assert corpus.full["train"][0].id == "full:train:0"


@pytest.mark.parametrize("mutation", ["merged", "count", "label", "shape", "text", "partition", "plus-mismatch"])
def test_reject_malformed_originals(originals, mutation):
    paths, contract = originals
    path = paths[1] if mutation == "plus-mismatch" else paths[0]
    value = json.loads(path.read_text())
    if mutation == "merged":
        value["train"].append(["out of scope", "oos"])
    elif mutation == "count":
        value["val"].pop()
    elif mutation == "label":
        value["test"][0][1] = "unknown"
    elif mutation == "shape":
        value["train"][0] = {"text": "not original format"}
    elif mutation == "text":
        value["train"][0][0] = " "
    elif mutation == "partition":
        del value["oos_val"]
    else:
        value["train"][0][0] = "different source text"
    path.write_bytes(canonical_json(value))
    with pytest.raises(ValueError):
        load_corpus(*paths, contract=contract)


def test_dmix_repeatability_hidden_labels_and_source_preservation(originals):
    corpus = tiny_corpus(originals)
    before = {name: file_hash(path) for name, path in corpus.source_paths.items()}
    first = build_dmix(corpus, 42, ind_count=4)
    second = build_dmix(corpus, 42, ind_count=4)
    assert first == second
    assert len(first.records) == 6
    assert len({row.id for row in first.records}) == 6
    assert sum(row.id.startswith("full:train:") for row in first.records) == 4
    assert sum(row.id.startswith("plus:oos_train:") for row in first.records) == 2
    assert all(set(asdict(row)) == {"id", "text"} for row in first.records)
    assert first.manifest["membership_sha256"] == content_hash(sorted(row.id for row in first.records))
    assert before == {name: file_hash(path) for name, path in corpus.source_paths.items()}
    assert build_dmix(corpus, 43, ind_count=4).manifest != first.manifest
    with pytest.raises(ValueError):
        build_dmix(corpus, ind_count=7)


def test_tokenizer_boundaries_padding_and_truncation():
    assert tokenize("CAFÉ, can't_2!") == ("café", ",", "can", "'", "t_2", "!")
    vocabulary = build_vocabulary(["b a café", "a"])
    assert vocabulary.tokens == ("<pad>", "<unk>", "<bos>", "<eos>", "a", "b", "café")
    assert build_vocabulary(["a", "café b a"]) == vocabulary
    encoded = encode_text("a missing", vocabulary, max_length=6)
    assert encoded.ids == (BOS, 4, UNK, EOS, PAD, PAD)
    assert encoded.length == 4
    truncated = encode_text("a b café missing", vocabulary, max_length=4)
    assert truncated.ids == (BOS, 4, 5, EOS)
    assert truncated.truncated_tokens == 2
    assert encode_text("", vocabulary, max_length=4).ids == (BOS, EOS, PAD, PAD)
    with pytest.raises(ValueError):
        encode_text("a", vocabulary, max_length=1)


def test_vocabularies_have_text_only_provenance(originals):
    corpus = tiny_corpus(originals)
    vocabs = build_vocabularies(corpus)
    assert "validationonly" not in vocabs.generator.tokens
    assert "plusoutlier" not in vocabs.generator.tokens
    assert "validationonly" in vocabs.classifier.tokens
    assert "oodtest" in vocabs.classifier.tokens
    assert vocabs.provenance["transductive_held_out_text"]
    assert "full:test" in vocabs.provenance["classifier_sources"]
    assert encode_text("validationonly", vocabs.generator).ids[1] == UNK
    for index, mapped in enumerate(vocabs.generator_to_classifier):
        assert vocabs.generator.tokens[index] == vocabs.classifier.tokens[mapped]
    train_only = build_vocabularies(corpus, profile="train-only")
    assert "validationonly" not in train_only.classifier.tokens
    assert not train_only.provenance["transductive_held_out_text"]


def test_prepared_bytes_manifests_and_collisions(originals, tmp_path):
    paths, contract = originals
    kwargs = dict(full_path=paths[0], plus_path=paths[1], contract=contract, ind_count=4, seed=17)
    before = [file_hash(path) for path in paths]
    first = prepare_data(tmp_path / "first", **kwargs)
    second = prepare_data(tmp_path / "second", **kwargs)
    assert first.manifest == second.manifest
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "second" / path.name).read_bytes()
    assert before == [file_hash(path) for path in paths]
    manifest = dict(first.manifest)
    manifest_hash = manifest.pop("manifest_sha256")
    assert content_hash(manifest) == manifest_hash
    for name, expected in first.manifest["files"].items():
        assert file_hash(tmp_path / "first" / name) == expected
    dmix = json.loads((tmp_path / "first/dmix.json").read_text())
    assert all(set(row) == {"id", "text"} for row in dmix)
    encoded = json.loads((tmp_path / "first/encoded-dmix-generator.json").read_text())
    assert all("target" not in row and "is_ood" not in row for row in encoded)
    with pytest.raises(FileExistsError):
        prepare_data(tmp_path / "first", **kwargs)
    assert first.manifest["embedding_state"] == "required-before-training"


def test_missing_and_wrong_dimension_embeddings(tmp_path):
    vocabulary = build_vocabulary(["known missing"])
    with pytest.raises(FileNotFoundError, match="glove.6B.100d.txt"):
        load_embeddings(vocabulary)
    path = tmp_path / "bad.txt"
    path.write_text("known 1 2\n")
    with pytest.raises(ValueError, match="100 dimensions"):
        load_embeddings(vocabulary, path)
    path.write_text("known " + " ".join(["nan"] * 100) + "\n")
    with pytest.raises(ValueError, match="finite"):
        load_embeddings(vocabulary, path)
    with pytest.raises(ValueError, match="cannot be combined"):
        load_embeddings(vocabulary, path, random_only=True)


def test_cuda_embedding_determinism_coverage_and_pad(tmp_path):
    """Run only in the parent's confirmed CUDA environment; never skip to CPU."""
    import torch

    vocabulary = build_vocabulary(["known missing"])
    path = tmp_path / "glove.6B.100d.txt"
    path.write_text("known " + " ".join(["0.25"] * 100) + "\n")
    first = load_embeddings(vocabulary, path, seed=8)
    second = load_embeddings(vocabulary, path, seed=8)
    assert first.weights.device.type == "cuda"
    assert first.weights.shape == (6, 100)
    assert torch.equal(first.weights, second.weights)
    assert torch.count_nonzero(first.weights[PAD]).item() == 0
    assert torch.all(first.weights[vocabulary.token_to_id["known"]] == 0.25).item()
    assert first.manifest["source_sha256"] == file_hash(path)
    assert first.manifest["coverage"] == 0.5
    random_first = load_embeddings(vocabulary, random_only=True, seed=8)
    random_second = load_embeddings(vocabulary, random_only=True, seed=8)
    assert torch.equal(random_first.weights, random_second.weights)
    assert random_first.manifest["profile"] == "random-only-smoke-non-paper"
    assert not torch.equal(first.weights, random_first.weights)
    with pytest.raises((ValueError, RuntimeError), match="(?i)cuda"):
        load_embeddings(vocabulary, random_only=True, device="cpu")


def test_real_original_counts_and_full_dmix():
    """Validate checked-out real files without downloads or tensor operations."""
    assert (ORIGINAL_DATA / "data_full.json").is_file(), "Initialize the CLINC submodule before acceptance"
    corpus = load_corpus()
    assert len(corpus.label_map) == 150
    assert {key: len(value) for key, value in corpus.full.items()} == {
        "train": 15000, "val": 3000, "test": 4500,
        "oos_train": 100, "oos_val": 100, "oos_test": 1000}
    mixture = build_dmix(corpus, seed=0)
    assert len(mixture.records) == 10250
    assert mixture.manifest["ind_count"] == 10000
    assert mixture.manifest["ood_count"] == 250
    assert all(row.id.startswith(("full:train:", "plus:oos_train:")) for row in mixture.records)
    assert corpus.source_hashes == {key: file_hash(path) for key, path in corpus.source_paths.items()}
