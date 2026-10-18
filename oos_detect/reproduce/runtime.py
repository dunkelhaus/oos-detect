"""CUDA selection and a short real-device verification."""

import platform
import time

import torch
from torch import nn


def require_cuda(device="cuda:0") -> torch.device:
    """Reject CPU selection and unavailable CUDA hardware."""
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("CUDA is required; CPU execution and fallback are not supported.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and the locked PyTorch wheel.")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} is unavailable.")
    return torch.device("cuda", index)


def diagnose(device="cuda:0") -> dict:
    """Check LSTM, convolution, and second-order input-gradient backward on CUDA."""
    selected = require_cuda(device)
    torch.cuda.synchronize(selected)
    torch.cuda.reset_peak_memory_stats(selected)
    start = time.perf_counter()
    # Construct parameters directly on CUDA, with no CPU model execution.
    lstm = nn.LSTM(8, 8, batch_first=True, device=selected)
    conv = nn.Conv1d(8, 8, 3, device=selected)
    inputs = torch.randn(2, 6, 8, device=selected, requires_grad=True)
    sequence, _ = lstm(inputs)
    conv(sequence.transpose(1, 2)).square().mean().backward()
    critic = nn.Sequential(nn.Linear(8, 16, device=selected), nn.LeakyReLU(),
                           nn.Linear(16, 1, device=selected))
    real = torch.randn(2, 8, device=selected)
    fake = torch.randn_like(real)
    alpha = torch.rand(2, 1, device=selected)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    gradient, = torch.autograd.grad(scores.sum(), interpolated, create_graph=True)
    penalty = (gradient.norm(2, dim=1) - 1).square().mean()
    penalty.backward()
    if not torch.isfinite(penalty) or not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for module in (lstm, conv, critic) for parameter in module.parameters()
    ):
        raise RuntimeError("CUDA diagnostic produced a non-finite loss or gradient.")
    torch.cuda.synchronize(selected)
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(selected),
        "gpu": torch.cuda.get_device_name(selected),
        "compute_capability": list(torch.cuda.get_device_capability(selected)),
        "checks": ["lstm_backward", "conv1d_backward", "gradient_penalty_second_order_backward"],
        "gp": penalty.item(), "seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected),
        "precision": "float32",
    }
