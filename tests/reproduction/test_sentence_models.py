"""CUDA-only acceptance checks for word-level POG models; no downloads."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pseudo_ood_generation.components.ae.pog_ae import (
    POGAutoEncoder, masked_token_cross_entropy,
)
from pseudo_ood_generation.components.auxilliary_classifier import (
    POGAuxilliaryClassifier, SentenceCNN,
)
from pseudo_ood_generation.components.sentence_utils import (
    annealed_temperature, expected_embeddings,
)


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "Sentence model tests require CUDA; no CPU fallback"
    torch.cuda.manual_seed_all(17)


def tiny_ae(**kwargs):
    return POGAutoEncoder(11, embedding_dim=8, code_dim=7, **kwargs)


def tiny_cnn(**kwargs):
    model = SentenceCNN(13, embedding_dim=8, num_classes=3,
                        out_channels=4, mlp_hidden_size=12, **kwargs)
    # Keep the tiny fixture out of a dead-ReLU region.
    with torch.no_grad():
        for conv in model.cnn:
            conv.bias.fill_(0.2)
        model.mlp1.bias.fill_(0.2)
        model.mlp2.bias.fill_(0.2)
    return model


def tokens():
    return torch.tensor([[2, 4, 5, 3, 0, 0], [2, 6, 3, 0, 0, 0]], device="cuda:0")


def mapping():
    # Deliberately not an identity: generator IDs must select classifier rows.
    return torch.tensor([0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 4], device="cuda:0")


def test_shared_table_shapes_padding_and_clean_code():
    ae = tiny_ae()
    assert ae.embedding is ae.encoder.embedding is ae.decoder.embedding
    assert all(p.is_cuda for p in ae.parameters())
    batch = tokens()
    output = ae(batch, add_noise=False)
    assert output["codes"].shape == (2, 7)
    assert output["logits"].shape == (2, 5, 11)
    assert output["probabilities"].shape == (2, 5, 11)
    assert torch.equal(output["mask"], batch[:, 1:].ne(0))
    torch.testing.assert_close(ae.encode(batch[:, :4]), ae.encode(batch))
    outputs, _ = ae.encoder.lstm(ae.embedding(batch))
    torch.testing.assert_close(output["codes"][0], outputs[0, 3])
    torch.testing.assert_close(output["codes"][1], outputs[1, 2])
    longer = F.pad(batch, (0, 5))
    torch.testing.assert_close(output["loss"], ae(longer, add_noise=False)["loss"])
    output["loss"].backward()
    assert ae.encoder.lstm.weight_ih_l0.grad.abs().sum() > 0
    assert ae.decoder.code_to_state.weight.grad.abs().sum() > 0
    assert ae.decoder.projection.weight.grad.abs().sum() > 0
    assert ae.embedding.weight.grad[0].count_nonzero() == 0


def test_masked_ce_and_all_padding():
    ae = tiny_ae()
    batch = torch.zeros(2, 4, dtype=torch.long, device="cuda:0")
    output = ae(batch, add_noise=False)
    assert output["codes"].count_nonzero() == 0
    assert output["loss"] == 0
    output["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ae.parameters())
    logits = torch.randn(2, 3, 11, device="cuda:0", requires_grad=True)
    targets = torch.tensor([[4, 3, 0], [3, 0, 0]], device="cuda:0")
    expected = F.cross_entropy(logits[targets.ne(0)], targets[targets.ne(0)])
    actual = masked_token_cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad[targets.eq(0)].count_nonzero() == 0


def test_tiny_no_noise_reconstruction_overfit():
    ae = POGAutoEncoder(8, embedding_dim=16, code_dim=16, noise_std=0)
    batch = torch.tensor([[2, 4, 5, 3], [2, 6, 7, 3]], device="cuda:0")
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.03, capturable=True)
    initial = ae(batch)["loss"].detach()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = ae(batch)["loss"]
        loss.backward()
        optimizer.step()
    result = ae(batch)
    assert result["loss"] < initial * 0.05
    assert torch.equal(result["logits"].argmax(-1), batch[:, 1:])
    assert ae.embedding.weight[0].count_nonzero() == 0


def test_gaussian_noise_seed_switch_and_rollout_modes():
    ae = tiny_ae()
    clean = ae.encode(tokens())
    rng = torch.Generator(device="cuda:0").manual_seed(91)
    noisy = ae.reconstruction_codes(clean, generator=rng)
    rng.manual_seed(91)
    expected = clean + torch.randn(clean.shape, device=clean.device, generator=rng)
    torch.testing.assert_close(noisy, expected)
    assert ae.reconstruction_codes(clean, add_noise=False) is clean
    assert not torch.equal(clean, noisy)
    for mode in ("soft", "sample", "greedy"):
        result = ae.rollout(clean, max_steps=5, mode=mode, generator=rng)
        assert result["logits"].shape == (2, 5, 11)
        assert torch.isfinite(result["logits"]).all()
        assert torch.isfinite(result["probabilities"]).all()
        assert result["mask"].dtype == torch.bool
        torch.testing.assert_close(result["probabilities"].sum(-1), torch.ones(2, 5, device="cuda:0"))
        assert result["probabilities"][..., [0, 2]].count_nonzero() == 0
    rng.manual_seed(8)
    first = ae.rollout(clean, 5, mode="sample", generator=rng)
    rng.manual_seed(8)
    second = ae.rollout(clean, 5, mode="sample", generator=rng)
    assert torch.equal(first["tokens"], second["tokens"])


def test_sampled_eos_mask_includes_eos_and_excludes_later_log_probs():
    ae = tiny_ae()
    with torch.no_grad():
        ae.decoder.projection.weight.zero_()
        ae.decoder.projection.bias.fill_(-100)
        ae.decoder.projection.bias[3] = 100
    result = ae.rollout(ae.encode(tokens()), 5, mode="sample")
    assert result["tokens"][:, 0].eq(3).all()
    assert result["tokens"][:, 1:].eq(0).all()
    assert result["mask"][:, 0].all()
    assert not result["mask"][:, 1:].any()
    assert result["log_probs"][:, 1:].count_nonzero() == 0
    assert torch.isfinite(result["sequence_log_probs"]).all()


def test_parallel_cnn_registration_shapes_padding_and_updates():
    model = tiny_cnn()
    assert isinstance(model.cnn, nn.ModuleList)
    assert [conv.kernel_size for conv in model.cnn] == [(2,), (3,), (4,), (5,)]
    assert len([m for m in model.modules() if isinstance(m, nn.Linear)]) == 3
    assert all(f"cnn.{i}.weight" in model.state_dict() for i in range(4))
    batch = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 10, 11, 3]], device="cuda:0")
    logits = model(batch)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(logits, model(F.pad(batch, (0, 6))))
    before = [conv.weight.detach().clone() for conv in model.cnn]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    F.cross_entropy(logits, torch.tensor([0, 1], device="cuda:0")).backward()
    optimizer.step()
    assert all(not torch.equal(old, conv.weight) for old, conv in zip(before, model.cnn))
    for length in (1, 2, 4):
        short = batch[:, :length]
        assert torch.isfinite(model(short)).all()
        torch.testing.assert_close(model(short), model(F.pad(short, (0, 7))))
    pads = torch.zeros(2, 3, dtype=torch.long, device="cuda:0")
    allpad_logits = model(pads)
    assert torch.isfinite(allpad_logits).all()
    allpad_logits.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.embedding.weight.grad[0].count_nonzero() == 0


def test_expected_embeddings_mapping_and_logit_gradients():
    model = tiny_cnn()
    logits = torch.randn(2, 6, 11, device="cuda:0", requires_grad=True)
    probabilities = (logits / 0.7).softmax(-1)
    mask = torch.ones(2, 6, dtype=torch.bool, device="cuda:0")
    expected = expected_embeddings(probabilities, model.embedding, mapping())
    torch.testing.assert_close(expected, probabilities @ model.embedding.weight[mapping()])
    result = model.forward_soft(logits, mask, 0.7, mapping())
    torch.testing.assert_close(result, model.forward_embeddings(expected, mask))
    result.square().sum().backward()
    assert logits.grad.abs().sum() > 0
    assert model.embedding.weight.grad[0].count_nonzero() == 0


def test_soft_autoregressive_feedback_reaches_previous_non_bos_embeddings():
    ae = tiny_ae()
    codes = torch.randn(2, 7, device="cuda:0", requires_grad=True)
    result = ae.rollout(codes, max_steps=5)
    result["logits"][:, -1].square().sum().backward()
    assert codes.grad.abs().sum() > 0
    # Only BOS is looked up directly. These gradients require q @ E feedback.
    assert ae.embedding.weight.grad[4:].abs().sum() > 0
    assert ae.embedding.weight.grad[0].count_nonzero() == 0


def test_auxiliary_only_optimizer_keeps_teacher_forced_ae_graph_without_updates():
    ae, classifier = tiny_ae(), tiny_cnn()
    ae_before = [p.detach().clone() for p in ae.parameters()]
    aux_before = [p.detach().clone() for p in classifier.parameters()]
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
    # Long enough for each kernel to have at least one valid target window.
    batch = torch.tensor([[2, 4, 5, 6, 7, 3], [2, 8, 9, 4, 5, 3]], device="cuda:0")
    reconstructed = ae(batch, add_noise=False)
    logits = classifier.forward_soft(reconstructed["logits"], reconstructed["mask"],
                                     temperature=0.8, token_mapping=mapping())
    F.cross_entropy(logits, torch.tensor([0, 1], device="cuda:0")).backward()
    assert ae.decoder.projection.weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(torch.equal(p, before) for p, before in zip(ae.parameters(), ae_before))
    assert any(not torch.equal(p, before) for p, before in zip(classifier.parameters(), aux_before))
    ae.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in ae.parameters())


@pytest.mark.parametrize("mode", ["soft", "sample"])
def test_frozen_decoder_classifier_preserve_generator_gradients(mode):
    ae, classifier = tiny_ae(), tiny_cnn()
    generator = nn.Linear(4, 7, device="cuda:0")
    frozen = list(ae.parameters()) + list(classifier.parameters())
    snapshots = [p.detach().clone() for p in frozen]
    for p in frozen:
        p.requires_grad_(False)
    codes = generator(torch.randn(3, 4, device="cuda:0"))
    result = ae.rollout(codes, max_steps=6, temperature=0.8, mode=mode)
    if mode == "soft":
        logits = classifier.forward_probabilities(result["probabilities"], result["mask"], mapping())
        log_probs = logits.log_softmax(-1)
        loss = (log_probs.exp() * log_probs).sum(-1).mean()
    else:
        mapped = mapping()[result["tokens"]]
        logits = classifier(mapped, result["mask"])
        # A nonconstant detached reward, as used by a policy-gradient trainer.
        reward = torch.tensor([0.5, -0.2, 1.0], device="cuda:0")
        loss = -(result["sequence_log_probs"] * reward).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert generator.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in frozen)
    assert all(torch.equal(p, old) for p, old in zip(frozen, snapshots))


def test_paper_dimensions_independent_tables_and_temperature_floor():
    ae = POGAutoEncoder(16)
    first = POGAuxilliaryClassifier(16)
    second = SentenceCNN(16)
    assert ae.embedding.embedding_dim == ae.encoder.lstm.hidden_size == 100
    assert first.mlp1.in_features == 512
    assert first.mlp1.out_features == first.mlp2.out_features == 512
    assert first.mlp3.out_features == 150
    assert first(tokens()).shape == (2, 150)
    assert first.embedding.weight.data_ptr() != second.embedding.weight.data_ptr()
    assert first.embedding.weight.data_ptr() != ae.embedding.weight.data_ptr()
    assert not torch.equal(first.cnn[0].weight, second.cnn[0].weight)
    assert annealed_temperature(0, 80) == 1
    assert annealed_temperature(80, 80) == pytest.approx(0.1)
    assert annealed_temperature(1000, 80) == pytest.approx(0.1)


def test_invalid_dimensions_tokens_modes_and_temperature():
    with pytest.raises(ValueError, match="CUDA"):
        POGAutoEncoder(11, device="cpu")
    with pytest.raises(ValueError, match="positive"):
        POGAutoEncoder(11, code_dim=0)
    with pytest.raises(ValueError, match="noise_std"):
        tiny_ae(noise_std=-1)
    ae = tiny_ae()
    bad = torch.tensor([[2, 0, 4]], device="cuda:0")
    with pytest.raises(ValueError, match="right padded"):
        ae.encode(bad)
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="Temperature"):
            ae.rollout(ae.encode(tokens()), temperature=temperature)
    with pytest.raises(ValueError, match="mode"):
        ae.rollout(ae.encode(tokens()), mode="straight-through")
    with pytest.raises(ValueError, match="Temperature"):
        annealed_temperature(0, 80, floor=0)


def test_pretrained_values_are_copied_and_pad_is_fixed_in_soft_paths():
    pretrained = torch.randn(13, 8, device="cuda:0")
    original = pretrained.clone()
    first = tiny_cnn(embeddings=pretrained)
    second = tiny_cnn(embeddings=pretrained)
    assert torch.equal(pretrained, original)
    torch.testing.assert_close(first.embedding.weight[1:], pretrained[1:])
    assert first.embedding.weight[0].count_nonzero() == 0
    assert first.embedding.weight.data_ptr() != second.embedding.weight.data_ptr()
    probabilities = torch.randn(2, 6, 13, device="cuda:0").softmax(-1)
    first.forward_probabilities(
        probabilities, torch.ones(2, 6, dtype=torch.bool, device="cuda:0")
    ).square().sum().backward()
    assert first.embedding.weight.grad[0].count_nonzero() == 0
    assert second.embedding.weight.grad is None
