"""Explicit command boundaries for paper reproduction."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="CLINC POG reproduction (CUDA required)")
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("runtime", help="verify CUDA and print runtime diagnostics")
    runtime.add_argument("--device", default="cuda:0", help="CUDA device; CPU is not supported")
    prepare = commands.add_parser("prepare", help="validate original CLINC sources and write text artifacts")
    prepare.add_argument("--output", type=Path, required=True, help="new artifact directory; must not exist")
    prepare.add_argument("--full", type=Path, help="original data_full.json; defaults to the dataset submodule")
    prepare.add_argument("--plus", type=Path, help="original data_oos_plus.json; defaults to the dataset submodule")
    prepare.add_argument("--seed", type=int, default=0, help="Dmix preparation seed")
    prepare.add_argument("--profile", choices=("paper", "train-only"), default="paper")
    prepare.add_argument("--max-length", type=int, default=64, help="BOS/EOS-inclusive token limit")
    prepare.add_argument("--dmix-ind-count", type=int, default=10000,
                         help="IND training records in Dmix; overrides are labeled non-paper")
    for name in ("train-generator", "generate", "train-classifier", "evaluate"):
        stage = commands.add_parser(name, help=f"explicit artifact-driven {name} stage")
        stage.add_argument("--prepared", type=Path, required=True)
        stage.add_argument("--output", type=Path, required=True, help="new output directory, or the same run with --resume")
        stage.add_argument("--seed", type=int, required=True, help="one declared campaign seed; never launches all seeds")
        stage.add_argument("--config", type=Path, help="resolved JSON config; cannot be combined with profile overrides")
        stage.add_argument("--profile", choices=("paper", "tiny-smoke-non-paper"))
        stage.add_argument("--variant", choices=("MSP", "Perturb", "Mix", "POG", "AEPOG"))
        stage.add_argument("--mode", choices=("pog", "aepog", "pog-no-noise", "pog-no-soft-token"))
        stage.add_argument("--device", help="CUDA device; defaults to config or cuda:0")
        stage.add_argument("--generator-epochs", type=int, help="paper requires 80; override only in non-paper profiles")
        stage.add_argument("--classifier-epochs", type=int, help="paper requires 30; override only in non-paper profiles")
        stage.add_argument("--seeds", type=int, nargs="+", help="declared campaign seeds; --seed selects exactly one")
        if name in {"train-generator", "train-classifier"}:
            stage.add_argument("--embeddings", type=Path, help="user-supplied GloVe 6B 100d; required for paper training")
            stage.add_argument("--resume", type=Path, help="trusted latest completed boundary from this run")
        if name in {"generate", "train-classifier"}:
            stage.add_argument("--generator-run", type=Path, required=name == "generate")
        if name == "train-classifier":
            stage.add_argument("--generated", type=Path, help="completed generated corpus directory")
        if name == "evaluate":
            stage.add_argument("--classifier-run", type=Path, required=True)
            stage.add_argument("--calibrate", action="store_true", help="calibrate a separate OOD threshold on validation only")
    aggregate = commands.add_parser("aggregate", help="aggregate explicit test summary files; report missing seeds")
    aggregate.add_argument("--reports", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True, help="new JSON report file")
    aggregate.add_argument("--expected-seeds", type=int, nargs="+", default=list(range(10)))
    smoke = commands.add_parser("smoke", help="bounded synthetic CUDA flow for all variants/ablations; explicitly non-paper")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "runtime":
        from .runtime import diagnose

        print(json.dumps(diagnose(args.device), indent=2))
    elif args.command == "prepare":
        from .data import prepare_data

        sources = {}
        if args.full is not None:
            sources["full_path"] = args.full
        if args.plus is not None:
            sources["plus_path"] = args.plus
        result = prepare_data(args.output, seed=args.seed, max_length=args.max_length,
                              profile=args.profile, ind_count=args.dmix_ind_count, **sources)
        print(json.dumps({"output": str(args.output), "manifest": result.manifest}, indent=2))
    else:
        from . import pipeline

        if args.command == "aggregate":
            result = pipeline.aggregate(reports=args.reports, output=args.output,
                                        expected_seeds=tuple(args.expected_seeds))
        elif args.command == "smoke":
            result = pipeline.smoke(output=args.output, device=args.device)
        else:
            from .config import load_config, paper_config, resolve_config, smoke_config

            if args.config is not None:
                if any(getattr(args, name) is not None for name in
                       ("profile", "variant", "mode", "device", "generator_epochs", "classifier_epochs", "seeds")):
                    parser.error("--config cannot be combined with profile, model, epoch, device, or seed-list overrides")
                config = load_config(args.config)
            else:
                factory = smoke_config if args.profile == "tiny-smoke-non-paper" else paper_config
                config = factory(args.variant or "POG", mode=args.mode, device=args.device or "cuda:0")
                overrides = {name: getattr(args, name) for name in ("generator_epochs", "classifier_epochs", "seeds")
                             if getattr(args, name) is not None}
                if overrides:
                    config = resolve_config({**config.to_dict(), **overrides})
            arguments = dict(prepared=args.prepared, output=args.output, config=config, seed=args.seed)
            if args.command in {"train-generator", "train-classifier"}:
                arguments.update(embedding_path=args.embeddings, resume=args.resume)
            if args.command in {"generate", "train-classifier"}:
                arguments["generator_run"] = args.generator_run
            if args.command == "train-classifier":
                arguments["generated"] = args.generated
            if args.command == "evaluate":
                arguments.update(classifier_run=args.classifier_run, calibrate=args.calibrate)
            operation = getattr(pipeline, args.command.replace("-", "_"))
            result = operation(**arguments)
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0
