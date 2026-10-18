"""Every tensor/model test runs on CUDA; no CPU fallback or skip."""

import pytest

from oos_detect.reproduce.runtime import diagnose, require_cuda


def test_cpu_selection_is_rejected():
    with pytest.raises(ValueError, match="CUDA is required"):
        require_cuda("cpu")


def test_target_card_diagnostic():
    result = diagnose()
    assert result["device"] == "cuda:0"
    assert result["peak_allocated_bytes"] > 0
    assert "gradient_penalty_second_order_backward" in result["checks"]


def test_diagnostic_initializes_cuda_in_a_fresh_process():
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "oos_detect.reproduce", "runtime", "--device", "cuda:0"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["peak_allocated_bytes"] > 0
