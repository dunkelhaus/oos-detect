"""Run a bounded, full-dimensional CUDA check. This is not a paper campaign."""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import platform
import time

import torch

from oos_detect.reproduce.classifier import (
    ClassifierConfig, build_final_classifier, classifier_objective, predict_confidence,
)
from oos_detect.reproduce.data import (
    build_dmix, build_vocabularies, encode_text, load_corpus,
)
from oos_detect.reproduce.runtime import diagnose, require_cuda
from pseudo_ood_generation.builders import build_model, build_trainer
from pseudo_ood_generation.trainer import POGSettings


def run_mode(mode, vocabularies, ind, targets, mixed, classifier_ind, device):
    """Run one generator cycle and one independent final CNN update."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    rng = torch.Generator(device=device).manual_seed(0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_model(
        len(vocabularies.generator.tokens), len(vocabularies.classifier.tokens),
        torch.tensor(vocabularies.generator_to_classifier, device=device),
        device=str(device),
    )
    trainer = build_trainer(model, POGSettings(mode=mode))
    assert all(p.is_cuda and p.dtype == torch.float32 for p in model.parameters())
    torch.cuda.synchronize(device)
    cycle_started = time.perf_counter()
    records = trainer.train_cycle(
        ind, targets, dmix_tokens=mixed if mode == "aepog" else None, rng=rng,
    )
    torch.cuda.synchronize(device)
    generator_seconds = time.perf_counter() - cycle_started
    assert len(records) == (11 if mode == "aepog" else 9)
    json.dumps(records, allow_nan=False)
    config = ClassifierConfig(
        paper=False, variant="AEPOG" if mode == "aepog" else "POG", device=str(device),
    )
    classifier = build_final_classifier(config, vocabularies.classifier)
    assert all(p.is_cuda and p.dtype == torch.float32 for p in classifier.parameters())
    assert not ({p.data_ptr() for p in classifier.parameters()}
                & {p.data_ptr() for p in model.auxiliary.parameters()})
    with torch.no_grad():
        codes = model.generator(model.generator.sample_latent(32, rng=rng))
        rollout = model.autoencoder.rollout(codes, max_steps=63, mode="sample", generator=rng)
        bos = torch.full((32, 1), 2, dtype=torch.long, device=device)
        entropy_tokens = torch.cat((bos, model.token_mapping[rollout["tokens"]]), dim=1)
    assert entropy_tokens.shape == (32, 64)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, capturable=True)
    before = classifier.mlp3.weight.detach().clone()
    torch.cuda.synchronize(device)
    started_classifier = time.perf_counter()
    losses = classifier_objective(classifier(classifier_ind), targets,
                                  classifier(entropy_tokens), alpha=1.0)
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all().item() for p in classifier.parameters()
               if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, classifier.mlp3.weight)
    torch.cuda.synchronize(device)
    classifier_seconds = time.perf_counter() - started_classifier
    predictions = predict_confidence(classifier, classifier_ind)
    assert predictions["logits"].shape == (32, 150)
    assert torch.isfinite(predictions["confidence"]).all().item()
    torch.cuda.synchronize(device)
    return {
        "mode": mode, "generator_seconds": generator_seconds,
        "classifier_seconds": classifier_seconds,
        "case_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "phase_counts": dict(Counter(row["phase"] for row in records)),
        "phases": records,
        "classifier": {key: value.item() for key, value in losses.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_cuda(args.device)
    # Reserve a new directory before any model update. Existing results stay intact.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    diagnostic = diagnose(args.device)
    corpus = load_corpus()
    vocabularies = build_vocabularies(corpus)
    rows = corpus.full["train"][:32]
    mixed_rows = build_dmix(corpus, seed=0).records[:32]

    def tokens(records, vocabulary):
        return torch.tensor([encode_text(row.text, vocabulary).ids for row in records],
                            dtype=torch.long, device=device)

    ind = tokens(rows, vocabularies.generator)
    mixed = tokens(mixed_rows, vocabularies.generator)
    classifier_ind = tokens(rows, vocabularies.classifier)
    targets = torch.tensor([row.target for row in rows], dtype=torch.long, device=device)
    cases = []
    for mode in ("pog", "aepog", "pog-no-noise", "pog-no-soft-token"):
        gc.collect()
        torch.cuda.empty_cache()
        cases.append(run_mode(mode, vocabularies, ind, targets, mixed, classifier_ind, device))
    report = {
        "profile": "non-paper-full-dimensional-batch32-smoke",
        "embeddings": "random-only; not a pretrained-vector experiment",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "batch_size": 32, "batch_reduction": False, "dtype": "float32",
        "max_length": 64, "rollout_steps": 63, "intent_outputs": 150,
        "vocabularies": {"generator": len(vocabularies.generator.tokens),
                         "classifier": len(vocabularies.classifier.tokens)},
        "diagnostic": diagnostic, "cases": cases,
        "seconds": time.perf_counter() - started,
        "paper_campaign_started": False,
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "acceptance.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
