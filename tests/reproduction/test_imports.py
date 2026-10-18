"""Import boundaries must not start a run or load obsolete dependencies."""

import importlib.metadata
import json
import subprocess
import sys


def test_supported_imports_and_legacy_entrypoint_are_safe():
    script = """
import sys, json
import oos_detect
import oos_detect.reproduce
import oos_detect.reproduce.cli
import oos_detect.train.__main__
import pseudo_ood_generation
import wgan_gp
for forbidden in ('allennlp', 'transformers', 'prefect', 'torchvision', 'imageio', 'wandb', 'pandas'):
    assert forbidden not in sys.modules, forbidden
print(json.dumps({'safe': True}))
"""
    result = subprocess.run([sys.executable, "-c", script], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == {"safe": True}


def test_help_does_not_require_model_execution():
    result = subprocess.run(
        [sys.executable, "-m", "oos_detect.reproduce", "--help"],
        check=True, text=True, capture_output=True,
    )
    assert "CUDA required" in result.stdout


def test_all_three_distributions_are_installed():
    for name in ("oos-detect", "pseudo-ood-generation", "wgan-gp"):
        assert importlib.metadata.version(name) == "0.1.0"
