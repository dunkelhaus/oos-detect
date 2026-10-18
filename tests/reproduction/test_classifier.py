"""CUDA final-classifier contracts. The parent runs these in Colab, not locally."""

from dataclasses import replace
import math

import pytest
import torch

from oos_detect.reproduce import classifier as api
from oos_detect.reproduce.data import (
    EmbeddingResult, Mixture, Record, UnlabeledRecord, build_vocabulary,
    content_hash,
)
from oos_detect.reproduce.state import Boundary, restore_checkpoint, save_checkpoint
from pseudo_ood_generation.components.auxilliary_classifier import SentenceCNN


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "Classifier model tests require CUDA; no CPU fallback"
    deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


@pytest.fixture
def vocabulary():
    return build_vocabulary(["book a flight now", "cancel my booking please", "unknown request"])


def config(variant="MSP", **changes):
    values = dict(variant=variant, paper=False, epochs=2, generator_epochs=3,
                  batch_size=2, embedding_dim=8, num_classes=2, out_channels=3,
                  mlp_hidden_size=8, max_length=8, seed=12)
    values.update(changes)
    return api.ClassifierConfig(**values)


def records(partition="train", count=3):
    texts = ("book a flight now", "cancel my booking please", "book my flight please")
    return tuple(Record("full", partition, index, texts[index % 3], False, index % 2)
                 for index in range(count))


def mixture(train, extra=1):
    rows = tuple(UnlabeledRecord(row.id, row.text) for row in train[:1]) + tuple(
        UnlabeledRecord(f"plus:oos_train:{i}", "unknown request") for i in range(extra))
    ids = [row.id for row in rows]
    return Mixture(rows, {"ids": ids, "labels_hidden": True, "ind_count": 1,
                          "ood_count": extra, "membership_sha256": content_hash(sorted(ids)),
                          "ordered_ids_sha256": content_hash(ids)})


def generated(vocabulary, variant="POG", count=5):
    rows = tuple(UnlabeledRecord(f"generated:{i}", "unknown request") for i in range(count))
    return api.GeneratedCorpus(rows, {"variant": variant, "epoch": 3, "epochs": 3,
                                      "checkpoint_sha256": "checkpoint-hash",
                                      "data_sha256": "data-hash",
                                      "vocabulary_sha256": vocabulary.sha256, "count": count})


def source(cfg, vocabulary, train):
    if cfg.variant == "Mix":
        kwargs = {"dmix": mixture(train, extra=4)}
    elif cfg.variant in {"POG", "AEPOG"}:
        kwargs = {"generated": generated(vocabulary, cfg.variant),
                  "data_sha256": "data-hash", "checkpoint_sha256": "checkpoint-hash"}
    else:
        kwargs = {}
    return api.make_entropy_source(cfg, vocabulary, train=train, **kwargs)


def test_paper_defaults_validation_and_pretrained_requirement(vocabulary):
    paper = api.ClassifierConfig()
    assert (paper.epochs, paper.generator_epochs, paper.alpha, paper.num_classes) == (30, 80, 1, 150)
    for changes in ({"alpha": 0.5}, {"epochs": 1}, {"embedding_dim": 8},
                    {"perturb_probability": 0.2}, {"num_classes": 151}):
        with pytest.raises(ValueError, match="Paper"):
            replace(paper, **changes)
    with pytest.raises(ValueError, match="pretrained"):
        api.build_final_classifier(paper, vocabulary)
    for changes in ({"device": "cpu"}, {"alpha": float("nan")}, {"seed": -1},
                    {"batch_size": 0}, {"variant": "wrong"}, {"learning_rate": 0}):
        with pytest.raises(ValueError):
            config(**changes)


def test_paper_cnn_is_fresh_not_auxiliary_weights(vocabulary):
    pretrained = torch.randn(len(vocabulary.tokens), 100, device="cuda:0")
    embedding = EmbeddingResult(pretrained, {"profile": "pretrained", "source_sha256": "hash",
                                            "vocabulary_sha256": vocabulary.sha256, "dimension": 100})
    auxiliary = SentenceCNN(len(vocabulary.tokens), embeddings=pretrained)
    with torch.no_grad():
        auxiliary.mlp3.bias.fill_(42)
    final = api.build_final_classifier(api.ClassifierConfig(seed=19), vocabulary, embedding)
    again = api.build_final_classifier(api.ClassifierConfig(seed=19), vocabulary, embedding)
    tokens = torch.tensor([[2, 4, 5, 6, 3]], device="cuda:0")
    assert final(tokens).shape == (1, 150)
    assert all(parameter.is_cuda for parameter in final.parameters())
    assert final.embedding.weight.data_ptr() != auxiliary.embedding.weight.data_ptr()
    assert final.embedding.weight.data_ptr() != pretrained.data_ptr()
    torch.testing.assert_close(final.embedding.weight[1:], pretrained[1:])
    assert not torch.equal(final.cnn[0].weight, auxiliary.cnn[0].weight)
    assert not torch.equal(final.mlp3.bias, auxiliary.mlp3.bias)
    for key, value in final.state_dict().items():
        torch.testing.assert_close(value, again.state_dict()[key], rtol=0, atol=0)
    before = auxiliary.embedding.weight.detach().clone()
    final(tokens).sum().backward()
    assert all(parameter.grad is None for parameter in auxiliary.parameters())
    assert torch.equal(before, auxiliary.embedding.weight)
    invalid = EmbeddingResult(pretrained, {**embedding.manifest, "vocabulary_sha256": "wrong"})
    with pytest.raises(ValueError, match="vocabulary"):
        api.build_final_classifier(api.ClassifierConfig(), vocabulary, invalid)


def test_entropy_sign_ce_separation_and_confidence(vocabulary):
    uniform = torch.zeros(3, 2, device="cuda:0")
    peaked = torch.tensor([[3.0, -1.0]], device="cuda:0", requires_grad=True)
    assert api.negative_entropy(uniform).item() == pytest.approx(-math.log(2))
    initial = api.negative_entropy(peaked)
    assert initial > api.negative_entropy(uniform)
    initial.backward()
    with torch.no_grad():
        improved = peaked - 0.2 * peaked.grad
    assert api.negative_entropy(improved) < initial
    ind = torch.tensor([[1.0, -1.0]], device="cuda:0", requires_grad=True)
    labels = torch.tensor([0], device="cuda:0")
    losses = api.classifier_objective(ind, labels, peaked, alpha=1)
    torch.testing.assert_close(losses["loss"], torch.nn.functional.cross_entropy(ind, labels) + initial)
    no_er = api.classifier_objective(ind, labels)
    torch.testing.assert_close(no_er["loss"], no_er["ce"])
    trainer = api.FinalClassifierTrainer(config(), vocabulary)
    tokens = torch.tensor([[2, 4, 5, 3]], device="cuda:0")
    trainer.model.train()
    predicted = api.predict_confidence(trainer.model, tokens)
    assert trainer.model.training
    assert not predicted["logits"].requires_grad
    confidence, intent = predicted["logits"].softmax(-1).max(-1)
    torch.testing.assert_close(predicted["confidence"], confidence)
    assert torch.equal(predicted["prediction"], intent)
    assert set(predicted) == {"logits", "prediction", "confidence"}


def test_perturb_seed_half_probability_uniform_and_special_preservation():
    tokens = torch.full((1024, 64), 4, dtype=torch.long, device="cuda:0")
    tokens[:, :4] = torch.arange(4, device="cuda:0")
    rng = torch.Generator(device="cuda:0").manual_seed(13)
    result = api.perturb_tokens(tokens, 9, generator=rng)
    rng.manual_seed(13)
    assert torch.equal(result, api.perturb_tokens(tokens, 9, generator=rng))
    assert torch.equal(tokens[:, :4], result[:, :4])
    content = result[:, 4:]
    # Original ID is an allowed replacement: observed changes ~= 0.5 * 4/5.
    assert content.ne(4).float().mean().item() == pytest.approx(0.4, abs=0.015)
    frequencies = torch.bincount(content.flatten(), minlength=9).float() / content.numel()
    assert frequencies[4].item() == pytest.approx(0.6, abs=0.015)
    for token in range(5, 9):
        assert frequencies[token].item() == pytest.approx(0.1, abs=0.01)
    assert torch.equal(tokens, api.perturb_tokens(tokens, 9, generator=rng, probability=0))
    assert tokens[:, 4:].eq(4).all()  # Never modify the caller's batch.


@pytest.mark.parametrize("variant", api.VARIANTS)
def test_sources_training_and_independent_cycling(vocabulary, variant):
    cfg = config(variant)
    train = records()
    er = source(cfg, vocabulary, train)
    if variant == "MSP":
        assert er is None
    else:
        assert all(type(row) is UnlabeledRecord and not hasattr(row, "target") for row in er.records)
    trainer = api.FinalClassifierTrainer(cfg, vocabulary)
    events = []
    result = trainer.fit(train, records("val", 2), source=er,
                         on_epoch=lambda event, owner: events.append((event, owner.completed_epoch)))
    assert len(result.history) == cfg.epochs
    assert result.final_epoch == 2
    assert result.selected_epoch in (1, 2)
    assert events[-1][0].final
    assert all(event.epoch == completed for event, completed in events)
    for metrics in result.history:
        assert metrics["ind_examples"] == (5 if variant in {"Mix", "POG", "AEPOG"} else 3)
        assert metrics["updates"] == (3 if variant in {"Mix", "POG", "AEPOG"} else 2)
        assert metrics["entropy_examples"] == (0 if variant == "MSP" else 3 if variant == "Perturb" else 5)
        assert metrics["ind_restarts"] == (1 if variant in {"Mix", "POG", "AEPOG"} else 0)
        assert math.isfinite(metrics["ce"])
        assert metrics["negative_entropy"] <= 0
    assert result.selected_state is not None and result.final_state is not None
    assert all(value.is_cuda for value in result.selected_state.values())
    assert result.final_state["mlp3.weight"].data_ptr() != trainer.model.mlp3.weight.data_ptr()


def test_shorter_entropy_loader_cycles_independently(vocabulary):
    cfg = config("Mix", epochs=1)
    train = records(count=7)
    er = api.make_entropy_source(cfg, vocabulary, train=train, dmix=mixture(train))
    result = api.FinalClassifierTrainer(cfg, vocabulary).fit(train, records("val"), source=er)
    metrics = result.history[0]
    assert (metrics["updates"], metrics["ind_examples"], metrics["entropy_examples"]) == (4, 7, 8)
    assert metrics["entropy_restarts"] == 3


def test_source_prerequisites_label_and_provenance_rejections(vocabulary):
    train = records()
    with pytest.raises(ValueError, match="no entropy"):
        api.make_entropy_source(config(), vocabulary, train=train, dmix=mixture(train))
    for variant in ("Mix", "POG", "AEPOG"):
        with pytest.raises(ValueError, match="requires"):
            api.make_entropy_source(config(variant), vocabulary, train=train)
    bad = mixture(train)
    for update in ({"labels_hidden": False}, {"membership_sha256": "wrong"}, {"ood_count": 999}):
        with pytest.raises(ValueError, match="membership"):
            api.make_entropy_source(config("Mix"), vocabulary, train=train,
                                    dmix=Mixture(bad.records, {**bad.manifest, **update}))
    leaked = (UnlabeledRecord("full:test:0", "unknown request"),)
    with pytest.raises(ValueError, match="membership"):
        api.make_entropy_source(config("Mix"), vocabulary, train=train,
                                dmix=Mixture(leaked, bad.manifest))
    corpus = generated(vocabulary)
    for field, value in (("variant", "AEPOG"), ("epoch", 2), ("epochs", 4),
                         ("vocabulary_sha256", "wrong"), ("checkpoint_sha256", "wrong"),
                         ("data_sha256", "wrong"), ("count", 999)):
        with pytest.raises(ValueError, match="provenance"):
            api.make_entropy_source(config("POG"), vocabulary, train=train,
                                    generated=api.GeneratedCorpus(corpus.records, {**corpus.manifest, field: value}),
                                    data_sha256="data-hash", checkpoint_sha256="checkpoint-hash")
    with pytest.raises(ValueError, match="no labels"):
        api.make_entropy_source(config("POG"), vocabulary, train=train,
                                generated=api.GeneratedCorpus(train, corpus.manifest),
                                data_sha256="data-hash", checkpoint_sha256="checkpoint-hash")
    trainer = api.FinalClassifierTrainer(config(), vocabulary)
    with pytest.raises(ValueError, match="test/OOD"):
        trainer.fit(records("test"), records("val"))
    with pytest.raises(ValueError, match="test/OOD"):
        trainer.fit(train, records("test"))
    with pytest.raises(ValueError, match="test/OOD"):
        trainer.fit((replace(train[0], is_ood=True, target=None),), records("val"))
    with pytest.raises(ValueError, match="no entropy"):
        trainer.fit(train, records("val"), source=source(config("Perturb"), vocabulary, train))


def test_final_generator_epoch_selection_ignores_metric_values():
    early = {"epoch": 1, "ood_accuracy": 1.0}
    final = {"epoch": 80, "ood_accuracy": 0.0}
    assert api.select_final_generator([early, final]) is final
    with pytest.raises(ValueError, match="Exactly one"):
        api.select_final_generator([early])
    with pytest.raises(ValueError, match="Exactly one"):
        api.select_final_generator([final, final])


def test_validation_selection_earliest_tie_and_both_weight_snapshots(vocabulary, monkeypatch):
    trainer = api.FinalClassifierTrainer(config(epochs=3), vocabulary)
    train, validation = records(), tuple(replace(row, target=0) for row in records("val", 2))
    # Validation alone selects the first epoch; later equal scores cannot replace it.
    monkeypatch.setattr(api, "predict_confidence", lambda model, tokens:
                        {"prediction": torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)})
    selected_snapshots, events = [], []

    def callback(event, owner):
        events.append(event)
        if event.selected:
            selected_snapshots.append(owner._snapshot())

    result = trainer.fit(train, validation, on_epoch=callback)
    assert result.selected_epoch == 1 and result.final_epoch == 3
    assert [event.selected for event in events] == [True, False, False]
    assert result.best_validation_accuracy == 1.0
    for key, value in result.selected_state.items():
        torch.testing.assert_close(value, selected_snapshots[0][key], rtol=0, atol=0)
    assert any(not torch.equal(result.selected_state[key], result.final_state[key])
               for key in result.selected_state)


def test_epoch_checkpoint_callback_resume_matches_uninterrupted(vocabulary, tmp_path):
    cfg = config("Perturb", epochs=3)
    train, validation = records(), records("val")
    er = source(cfg, vocabulary, train)
    uninterrupted = api.FinalClassifierTrainer(cfg, vocabulary)
    expected = uninterrupted.fit(train, validation, source=er)
    interrupted = api.FinalClassifierTrainer(cfg, vocabulary)
    checkpoint = tmp_path / "epoch-1.pt"

    class StopAtBoundary(Exception):
        pass

    def save_and_stop(event, owner):
        save_checkpoint(checkpoint, models={"classifier": owner.model},
                        optimizers={"classifier": owner.optimizer}, generators=owner.generators,
                        boundary=Boundary(event.epoch, 0, "classifier_epoch"),
                        compatibility_hash="compatible", temperature=1, policy_baseline=None,
                        extra={"classifier_selection": owner.selection_state()})
        raise StopAtBoundary

    with pytest.raises(StopAtBoundary):
        interrupted.fit(train, validation, source=er, on_epoch=save_and_stop)
    resumed = api.FinalClassifierTrainer(cfg, vocabulary)
    restored = restore_checkpoint(checkpoint, models={"classifier": resumed.model},
                                  optimizers={"classifier": resumed.optimizer},
                                  generators=resumed.generators, compatibility_hash="compatible")
    resumed.restore_selection(restored["extra"]["classifier_selection"])
    actual = resumed.fit(train, validation, source=er)
    assert [row["epoch"] for row in actual.history] == [2, 3]
    assert actual.selected_epoch == expected.selected_epoch
    assert actual.best_validation_accuracy == expected.best_validation_accuracy
    for section in ("selected_state", "final_state"):
        for key, value in getattr(expected, section).items():
            torch.testing.assert_close(value, getattr(actual, section)[key], rtol=0, atol=0)
    assert all(not isinstance(value, torch.Tensor) or value.is_cuda
               for state in resumed.optimizer.state.values() for value in state.values())
