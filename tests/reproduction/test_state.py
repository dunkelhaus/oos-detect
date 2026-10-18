import json

import pytest
import torch

from oos_detect.reproduce.runtime import require_cuda
from oos_detect.reproduce.state import (
    Boundary, create_run, derived_seed, restore_checkpoint, save_checkpoint, seed_run,
)


def test_independent_seed_streams():
    assert derived_seed(0, "prepare") == derived_seed(0, "prepare")
    assert derived_seed(0, "prepare") != derived_seed(0, "generation")


def test_completed_boundary_continuation(tmp_path):
    device = require_cuda()
    streams = seed_run(7)
    model = torch.nn.Linear(3, 2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def step():
        inputs = torch.randn(4, 3, device=device, generator=streams["loader"])
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    step()
    checkpoint = tmp_path / "boundary.pt"
    arguments = dict(models={"model": model}, optimizers={"model": optimizer},
                     generators=streams, compatibility_hash="same-inputs")
    save_checkpoint(checkpoint, **arguments, boundary=Boundary(0, 1, "reconstruction"),
                    temperature=0.5, policy_baseline=0.1)
    expected_loss = step()
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    restored = restore_checkpoint(checkpoint, **arguments)
    assert restored["boundary"] == Boundary(0, 1, "reconstruction")
    assert restored["temperature"] == 0.5
    assert restored["policy_baseline"] == 0.1
    assert torch.equal(step(), expected_loss)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, expected[name])
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, **arguments, boundary=Boundary(0, 2, "critic"),
                        temperature=0.5, policy_baseline=0.1)
    arguments["compatibility_hash"] = "different-data"
    with pytest.raises(ValueError, match="does not match"):
        restore_checkpoint(checkpoint, **arguments)


def test_manifest_collision_preserves_outputs(tmp_path):
    repository = __import__("pathlib").Path(__file__).resolve().parents[2]
    directory = tmp_path / "run"
    create_run(directory, {"seed": 0, "profile": "smoke"}, "data", "embedding", repository)
    original = (directory / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run(directory, {"seed": 1}, "data", "embedding", repository)
    assert (directory / "manifest.json").read_bytes() == original
    manifest = json.loads(original)
    assert set(manifest["source"]) == {"root", "pog", "gan"}
