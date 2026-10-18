"""CUDA-only phase ownership and POG mechanism acceptance tests."""

import copy
import math

import pytest
import torch

from pseudo_ood_generation.builders import build_model, build_trainer
from pseudo_ood_generation.trainer import POGSettings
import pseudo_ood_generation.trainer as trainer_module


@pytest.fixture
def fixture():
    assert torch.cuda.is_available(), "POG acceptance tests require CUDA; no CPU fallback"
    torch.manual_seed(731)
    torch.cuda.manual_seed_all(731)
    model = build_model(11, 13, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12],
                        embedding_dim=6, code_dim=5, noise_dim=4, gan_hidden_dim=12,
                        gan_layers=3, num_classes=3, out_channels=4, mlp_hidden_size=9)
    # Keep this small acceptance fixture out of a dead-ReLU region.
    with torch.no_grad():
        for conv in model.auxiliary.cnn:
            conv.bias.fill_(0.2)
        model.auxiliary.mlp1.bias.fill_(0.2)
        model.auxiliary.mlp2.bias.fill_(0.2)
    tokens = torch.tensor([[2, 4, 5, 6, 7, 8, 3, 0], [2, 5, 7, 8, 9, 3, 0, 0]], device="cuda")
    labels = torch.tensor([0, 2], device="cuda")
    return model, tokens, labels


def settings(mode="pog", **kwargs):
    return POGSettings(mode=mode, critic_iterations=2, max_steps=7, epochs=3, **kwargs)


def snapshots(model):
    return {name: [p.detach().clone() for p in module.parameters()]
            for name, module in model.phase_modules.items()}


def assert_owner(model, before, owner):
    for name, module in model.phase_modules.items():
        changed = any(not torch.equal(old, new) for old, new in zip(before[name], module.parameters()))
        assert changed == (name == owner), f"Unexpected update in {name}; phase owner is {owner}"
    assert all(p.grad is None for p in model.parameters()), "Phase left stale gradients"


@pytest.mark.parametrize("phase,owner", [("reconstruction", "ae"), ("auxiliary", "auxiliary"),
                                         ("critic", "critic"), ("generator_wasserstein", "generator"),
                                         ("generator_entropy", "generator")])
def test_phase_parameter_isolation(fixture, phase, owner):
    model, tokens, labels = fixture
    trainer = build_trainer(model, settings())
    before = snapshots(model)
    if phase == "auxiliary":
        record = trainer.auxiliary_step(tokens, labels)
    elif phase in {"generator_entropy", "generator_wasserstein"}:
        record = getattr(trainer, phase + "_step")(tokens.shape[0])
    else:
        record = getattr(trainer, phase + "_step")(tokens)
    assert_owner(model, before, owner)
    assert record["gradient_norm"] > 0
    assert record["phase"] == phase
    assert record["samples"] == tokens.shape[0]
    assert all(math.isfinite(value) for key, value in record.items() if key != "phase")
    if phase == "generator_entropy":
        assert record["entropy_loss"] < 0
        assert record["entropy"] > 0


def test_auxiliary_retains_then_discards_ae_gradients(fixture):
    model, tokens, labels = fixture
    seen = []
    handle = model.autoencoder.decoder.projection.weight.register_hook(
        lambda grad: seen.append(grad.detach().norm().item()))
    trainer = build_trainer(model, settings())
    before = snapshots(model)
    try:
        trainer.auxiliary_step(tokens, labels)
    finally:
        handle.remove()
    assert seen and max(seen) > 0
    assert_owner(model, before, "auxiliary")


def test_critic_receives_clean_detached_codes(fixture, monkeypatch):
    model, tokens, _ = fixture
    trainer = build_trainer(model, settings())
    expected = model.autoencoder.encode(tokens).detach()
    original = trainer_module.critic_step
    seen = []

    def capture(generator, critic, optimizer, real_codes, **kwargs):
        assert not real_codes.requires_grad
        torch.testing.assert_close(real_codes, expected)
        seen.append(True)
        return original(generator, critic, optimizer, real_codes, **kwargs)

    monkeypatch.setattr(trainer_module, "critic_step", capture)
    trainer.critic_step(tokens)
    assert seen


@pytest.mark.parametrize("mode", ["pog", "aepog", "pog-no-noise", "pog-no-soft-token"])
def test_cycles_counts_and_mechanisms(fixture, mode, monkeypatch):
    model, tokens, labels = fixture
    logs, noise_flags, rollout_modes = [], [], []
    trainer = build_trainer(model, settings(mode), log_callback=logs.append)
    original_forward, original_rollout = model.autoencoder.forward, model.autoencoder.rollout

    def capture_forward(*args, **kwargs):
        noise_flags.append(kwargs["add_noise"])
        return original_forward(*args, **kwargs)

    def capture_rollout(*args, **kwargs):
        rollout_modes.append(kwargs["mode"])
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(model.autoencoder, "forward", capture_forward)
    monkeypatch.setattr(model.autoencoder, "rollout", capture_rollout)
    records = trainer.train_cycle(tokens, labels, dmix_tokens=tokens.flip(0) if mode == "aepog" else None)
    assert records == logs
    expected = {"reconstruction": 1, "auxiliary": 1, "critic": 2,
                "generator_wasserstein": 1, "generator_entropy": 1}
    if mode == "aepog":
        expected.update(dmix_reconstruction=1, dmix_critic=1)
    assert trainer.update_counts == expected
    assert trainer.sample_counts == {key: 2 * value for key, value in expected.items()}
    assert noise_flags and all(flag == (mode != "pog-no-noise") for flag in noise_flags)
    assert rollout_modes == ["sample" if mode == "pog-no-soft-token" else "soft"]
    assert all(p.grad is None for p in model.parameters())
    assert trainer.baseline_updates == (1 if mode == "pog-no-soft-token" else 0)


@pytest.mark.parametrize("mode", ["pog", "pog-no-noise", "pog-no-soft-token"])
def test_dmix_is_rejected_before_pog_updates(fixture, mode):
    model, tokens, labels = fixture
    trainer = build_trainer(model, settings(mode))
    before = snapshots(model)
    with pytest.raises(ValueError, match="Dmix"):
        trainer.train_cycle(tokens, labels, dmix_tokens=tokens)
    with pytest.raises(ValueError, match="Dmix"):
        trainer.reconstruction_step(tokens, dmix=True)
    with pytest.raises(ValueError, match="Dmix"):
        trainer.critic_step(tokens, dmix=True)
    assert not trainer.update_counts
    for name, module in model.phase_modules.items():
        assert all(torch.equal(old, new) for old, new in zip(before[name], module.parameters()))


def test_aepog_accepts_only_unlabeled_tensor(fixture):
    model, tokens, labels = fixture
    trainer = build_trainer(model, settings("aepog"))
    with pytest.raises(ValueError, match="Dmix"):
        trainer.train_cycle(tokens, labels)
    with pytest.raises(ValueError, match="unlabeled"):
        trainer.train_cycle(tokens, labels, dmix_tokens={"tokens": tokens, "labels": labels})
    assert not trainer.update_counts


def test_reinforce_isolated_checkpointed_and_repeatable(fixture):
    model, tokens, _ = fixture
    trainer = build_trainer(model, settings("pog-no-soft-token"))
    before = snapshots(model)
    result = trainer.generator_entropy_step(2)
    assert result["gradient_norm"] > 0
    assert_owner(model, before, "generator")
    assert trainer.baseline > 0
    assert trainer.baseline_updates == 1
    model_state = copy.deepcopy(model.state_dict())
    trainer_state = copy.deepcopy(trainer.state_dict())
    rng = torch.Generator(device="cuda").manual_seed(55)
    rng_state = rng.get_state()
    expected = trainer.generator_entropy_step(tokens.shape[0], rng=rng)
    expected_parameters = snapshots(model)
    model.load_state_dict(model_state)
    resumed = build_trainer(model, settings("pog-no-soft-token"))
    resumed.load_state_dict(trainer_state)
    assert resumed.baseline == trainer_state["baseline"]
    rng.set_state(rng_state)
    actual = resumed.generator_entropy_step(tokens.shape[0], rng=rng)
    assert actual == pytest.approx(expected)
    for name, module in model.phase_modules.items():
        for old, new in zip(expected_parameters[name], module.parameters()):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)


def test_schedule_defaults_and_incompatible_resume(fixture):
    model, _, _ = fixture
    defaults = POGSettings()
    assert (defaults.batch_size, defaults.epochs, defaults.critic_iterations) == (32, 80, 5)
    assert (defaults.generator_lr, defaults.critic_lr, defaults.gan_betas) == (1e-4, 1e-4, (0.0, 0.9))
    trainer = build_trainer(model, settings())
    assert trainer.temperature == 1
    trainer.set_epoch(2)
    assert trainer.temperature == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Incompatible"):
        build_trainer(model, settings("pog-no-noise")).load_state_dict(trainer.state_dict())


def test_nonfinite_failure_clears_gradients(fixture):
    model, tokens, _ = fixture
    trainer = build_trainer(model, settings())
    with torch.no_grad():
        next(model.autoencoder.parameters()).flatten()[0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        trainer.reconstruction_step(tokens)
    assert not trainer.update_counts
    assert all(p.grad is None for p in model.parameters())


def test_optimizer_ownership_rejects_shared_owner(fixture):
    model, _, _ = fixture
    trainer = build_trainer(model, settings())
    optimizers = dict(trainer.optimizers)
    optimizers["auxiliary"] = optimizers["ae"]
    with pytest.raises(ValueError, match="ownership"):
        build_trainer(model, settings(), optimizers=optimizers)
