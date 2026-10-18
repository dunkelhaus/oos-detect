"""CUDA acceptance tests for the installed, fork-owned latent GAN."""

import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

_modules_before_gan_import = set(sys.modules)
import wgan_gp
from wgan_gp import (
    Critic, Generator, Trainer, critic_loss, critic_step, frozen_parameters,
    generator_loss, generator_step, gradient_penalty,
)
_gan_imported_modules = set(sys.modules) - _modules_before_gan_import


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "These model tests require CUDA; no CPU fallback"
    torch.cuda.manual_seed_all(7)


def models():
    generator = Generator(noise_dim=3, code_dim=2, hidden_dim=8)
    critic = Critic(code_dim=2, hidden_dim=8)
    return generator, critic


def optimizers(generator, critic):
    return (torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.0, 0.9), capturable=True),
            torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.9), capturable=True))


def snapshot(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def unchanged(module, before):
    return all(torch.equal(parameter, saved)
               for parameter, saved in zip(module.parameters(), before))


def test_installed_package_ownership_and_import_boundary():
    path = Path(wgan_gp.__file__).resolve()
    assert path.parent.name == "wgan_gp"
    assert path.parent.parent.name == "wgan-gp"
    assert not any(name.split(".")[0] in {"torchvision", "imageio", "allennlp", "wandb"}
                   for name in _gan_imported_modules)
    assert wgan_gp.Discriminator is Critic


def test_paper_defaults_and_vector_shapes():
    generator, critic = Generator(), Critic()
    for model in (generator, critic):
        affine = [module for module in model.modules() if isinstance(module, nn.Linear)]
        assert len(affine) == 4
        assert [layer.out_features for layer in affine[:-1]] == [512, 512, 512]
        assert len([module for module in model.modules() if isinstance(module, nn.LeakyReLU)]) == 3
        assert not any(isinstance(module, (nn.Sigmoid, nn.Conv2d, nn.BatchNorm1d))
                       for module in model.modules())
        assert all(parameter.is_cuda for parameter in model.parameters())
    noise = generator.sample_latent(3)
    assert noise.shape == (3, 100) and noise.is_cuda
    codes = generator(noise)
    assert codes.shape == (3, 100)
    assert critic(codes).shape == (3, 1)
    with torch.no_grad():
        critic.network[-1].weight.zero_()
        critic.network[-1].bias.fill_(3)
    assert (critic(codes) > 1).all()
    with torch.no_grad():
        critic.network[-1].bias.fill_(-3)
    assert (critic(codes) < 0).all()


@pytest.mark.parametrize("kwargs", [
    {"noise_dim": 0}, {"code_dim": -1}, {"hidden_dim": 0},
    {"num_layers": 1}, {"num_layers": True}, {"negative_slope": float("nan")},
])
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        Generator(**kwargs)


def test_invalid_vectors_and_cpu_selection():
    with pytest.raises(ValueError, match="CUDA"):
        Generator(device="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        Critic(device="cpu")
    generator, critic = models()
    for invalid in (torch.randn(3, 4, device="cuda"),
                    torch.randn(3, 1, 3, device="cuda"),
                    torch.empty(0, 3, device="cuda")):
        with pytest.raises(ValueError):
            generator(invalid)
    with pytest.raises(ValueError):
        critic(torch.randn(2, 3, device="cuda"))
    with pytest.raises(ValueError):
        generator.sample_latent(0)


def test_analytic_penalty_and_second_order_gradients(monkeypatch):
    critic = nn.Linear(2, 1, bias=False, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 4.0]], device="cuda", dtype=torch.float64))
    real = torch.randn(5, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    fake = torch.randn(5, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    draws = []
    original_rand = torch.rand

    def record_rand(*shape, **kwargs):
        draws.append(shape)
        return original_rand(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", record_rand)
    seen = []
    handle = critic.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
    try:
        penalty = gradient_penalty(critic, real, fake)
    finally:
        handle.remove()
    torch.testing.assert_close(penalty, real.new_tensor(16.0))
    penalty.backward()
    torch.testing.assert_close(critic.weight.grad, real.new_tensor([[4.8, 6.4]]))
    assert real.grad is None and fake.grad is None
    assert seen == [torch.Size([5, 2])]
    assert draws == [(5, 1)]
    assert torch.isfinite(critic.weight.grad).all()


def test_unit_gradient_penalty_and_interpolation_shape():
    critic = nn.Linear(2, 1, bias=False, device="cuda")
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]], device="cuda"))
    real = torch.randn(4, 2, device="cuda")
    fake = torch.randn_like(real)
    alpha = torch.rand(4, 1, device="cuda")
    torch.testing.assert_close(gradient_penalty(critic, real, fake, alpha=alpha),
                               real.new_tensor(0.0))
    for invalid in (torch.rand(4, 2, device="cuda"), real.new_full((4, 1), 2.0)):
        with pytest.raises(ValueError):
            gradient_penalty(critic, real, fake, alpha=invalid)
    with pytest.raises(ValueError):
        gradient_penalty(critic, real, fake[:2])


def test_wasserstein_loss_signs():
    real = torch.tensor([[2.0], [4.0]], device="cuda")
    fake = torch.tensor([[-1.0], [1.0]], device="cuda")
    penalty = real.new_tensor(2.0)
    torch.testing.assert_close(critic_loss(real, fake, penalty, 10), real.new_tensor(17))
    torch.testing.assert_close(generator_loss(fake), real.new_tensor(0))
    with pytest.raises(ValueError):
        critic_loss(real, fake, penalty, 0)


def test_independent_updates_and_encoder_isolation():
    generator, critic = models()
    gen_optimizer, critic_optimizer = optimizers(generator, critic)
    encoder = nn.Linear(4, 2, device="cuda")
    real = encoder(torch.randn(6, 4, device="cuda"))
    generator_before, critic_before, encoder_before = map(snapshot, (generator, critic, encoder))
    critic_diagnostics = critic_step(generator, critic, critic_optimizer, real)
    assert unchanged(generator, generator_before)
    assert unchanged(encoder, encoder_before)
    assert not unchanged(critic, critic_before)
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    critic_before = snapshot(critic)
    generator_diagnostics = generator_step(generator, critic, gen_optimizer, 6)
    assert unchanged(critic, critic_before)
    assert unchanged(encoder, encoder_before)
    assert not unchanged(generator, generator_before)
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert generator_diagnostics["gradient_norm"] > 0
    assert all(math.isfinite(value) for diagnostics in (critic_diagnostics, generator_diagnostics)
               for value in diagnostics.values())
    assert all(parameter.requires_grad for parameter in critic.parameters())


def test_frozen_guard_preserves_flags_and_input_gradient_on_exception():
    critic = Critic(code_dim=2, hidden_dim=8)
    next(critic.parameters()).requires_grad_(False)
    flags = [parameter.requires_grad for parameter in critic.parameters()]
    inputs = torch.randn(4, 2, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with frozen_parameters(critic):
            assert not any(parameter.requires_grad for parameter in critic.parameters())
            critic(inputs).sum().backward()
            raise RuntimeError("fixture failure")
    assert [parameter.requires_grad for parameter in critic.parameters()] == flags
    assert inputs.grad is not None and inputs.grad.norm() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_configurable_counts_and_standalone_example():
    generator, critic = models()
    gen_optimizer, critic_optimizer = optimizers(generator, critic)
    trainer = Trainer(generator, critic, gen_optimizer, critic_optimizer,
                      critic_iterations=2, generator_iterations=3)
    result = trainer.train_cycle(torch.randn(4, 2, device="cuda"))
    assert len(result["critic"]) == trainer.critic_updates == 2
    assert len(result["generator"]) == trainer.generator_updates == 3
    assert all(math.isfinite(value) for phase in result.values()
               for update in phase for value in update.values())
    for kwargs in ({"critic_iterations": 0}, {"generator_iterations": -1}):
        with pytest.raises(ValueError):
            Trainer(generator, critic, gen_optimizer, critic_optimizer, **kwargs)


def test_nonfinite_and_wrong_optimizer_fail_before_updates():
    generator, critic = models()
    gen_optimizer, critic_optimizer = optimizers(generator, critic)
    before = snapshot(critic)
    real = torch.full((4, 2), float("nan"), device="cuda")
    with pytest.raises(FloatingPointError):
        critic_step(generator, critic, critic_optimizer, real)
    assert unchanged(critic, before)
    with pytest.raises(ValueError, match="Optimizer"):
        critic_step(generator, critic, gen_optimizer, torch.randn(4, 2, device="cuda"))


def test_cuda_rng_repeats_noise_without_hardcoded_index():
    generator, _ = models()
    device = next(generator.parameters()).device
    rng = torch.Generator(device=device).manual_seed(41)
    first = generator.sample_latent(4, rng=rng)
    rng.manual_seed(41)
    torch.testing.assert_close(first, generator.sample_latent(4, rng=rng))
    assert first.device == device
