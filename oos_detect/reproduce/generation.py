"""Independent CUDA text export from trusted state.py checkpoints."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .data import BOS, EOS, PAD, UNK, UnlabeledRecord, Vocabulary, content_hash
from .runtime import require_cuda
from .state import canonical_json, derived_seed, file_hash, fingerprint, write_json


MODES = {"pog": "POG", "aepog": "AEPOG", "pog-no-noise": "POG-no-noise",
         "pog-no-soft-token": "POG-no-soft-token"}


@dataclass(frozen=True)
class GenerationConfig:
    """Small exports require paper=False; length includes the initial BOS."""

    seed: int
    paper: bool = True
    count: int = 15000
    batch_size: int = 32
    max_length: int = 64
    max_empty_retries: int = 10
    device: str = "cuda:0"

    def __post_init__(self):
        if type(self.paper) is not bool:
            raise ValueError("paper must be a boolean")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1)")
        if any(type(n) is not int or n < 1 for n in
               (self.count, self.batch_size, self.max_length)) or self.max_length < 2:
            raise ValueError("Counts must be positive; max_length must include BOS and one output")
        if type(self.max_empty_retries) is not int or self.max_empty_retries < 0:
            raise ValueError("max_empty_retries must be a nonnegative integer")
        if torch.device(self.device).type != "cuda":
            raise ValueError("Generation requires CUDA; no CPU fallback")
        if self.paper and (self.count != 15000 or self.max_length != 64):
            raise ValueError("Paper generation requires 15000 records and a 64-token cap")


def checkpoint_generation_metadata(*, mode, completed_epochs, configured_epochs,
                                   data_sha256, generator_vocabulary, classifier_vocabulary):
    """Put this result in save_checkpoint(extra={\"generation\": result}).

    Call only after a complete epoch, never after a partial training cycle.
    Epoch counts are one-based counts, not the trainer's zero-based epoch index.
    """
    if mode not in MODES:
        raise ValueError("Unknown generator mode")
    if (type(completed_epochs) is not int or type(configured_epochs) is not int
            or not 1 <= completed_epochs <= configured_epochs or not data_sha256):
        raise ValueError("Require valid completed/configured epoch counts and data hash")
    return {"mode": mode, "variant": MODES[mode], "epoch": completed_epochs,
            "epochs": configured_epochs, "selection": "final-configured-epoch",
            "data_sha256": data_sha256,
            "generator_vocabulary_sha256": generator_vocabulary.sha256,
            "vocabulary_sha256": classifier_vocabulary.sha256}


def _prepared_identity(manifest, generator_vocabulary, classifier_vocabulary):
    unsigned = dict(manifest)
    identity = unsigned.pop("manifest_sha256", None)
    if not identity or content_hash(unsigned) != identity:
        raise ValueError("Prepared manifest hash mismatch")
    if manifest.get("vocabulary_hashes") != {
            "generator": generator_vocabulary.sha256, "classifier": classifier_vocabulary.sha256}:
        raise ValueError("Prepared vocabulary hashes do not match")
    return identity


def load_generation_checkpoint(path, *, model, run_manifest, prepared_manifest,
                               generator_vocabulary, classifier_vocabulary,
                               configured_epochs=80, device="cuda:0", model_key="pog"):
    """Load model weights only; do not restore training RNG or optimizer state.

    Only load trusted local files: state.py includes pickled NumPy/Python state.
    model_key=None loads state.py models saved with model.phase_modules instead.
    """
    data_hash = _prepared_identity(prepared_manifest, generator_vocabulary, classifier_vocabulary)
    expected = fingerprint({key: run_manifest[key]
                            for key in ("config", "data_hash", "embedding_hash")})
    if (run_manifest.get("compatibility_hash") != expected
            or run_manifest.get("data_hash") != data_hash):
        raise ValueError("Run compatibility or data hash mismatch")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing generator checkpoint: {path}; train it explicitly first")
    selected = require_cuda(device)
    checkpoint_hash = file_hash(path)
    state = torch.load(path, map_location=selected, weights_only=False)
    if file_hash(path) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    if state.get("format_version") != 1 or state.get("compatibility_hash") != expected:
        raise ValueError("Checkpoint compatibility hash mismatch")
    metadata = state.get("extra", {}).get("generation", {})
    mode = metadata.get("mode")
    if mode not in MODES:
        raise ValueError("Checkpoint needs explicit extra.generation mode and epoch provenance")
    required = checkpoint_generation_metadata(
        mode=mode, completed_epochs=configured_epochs, configured_epochs=configured_epochs,
        data_sha256=data_hash, generator_vocabulary=generator_vocabulary,
        classifier_vocabulary=classifier_vocabulary)
    if metadata != required:
        raise ValueError("Checkpoint must match data/vocabulary and the final configured epoch")
    if any(p.device != selected for p in model.parameters()):
        raise ValueError("Model parameters must use the selected CUDA device")
    if (model.autoencoder.embedding.num_embeddings != len(generator_vocabulary.tokens)
            or model.auxiliary.embedding.num_embeddings != len(classifier_vocabulary.tokens)
            or model.autoencoder.pad_id != PAD or model.autoencoder.decoder.bos_id != BOS
            or model.autoencoder.decoder.eos_id != EOS):
        raise ValueError("Model vocabulary sizes or boundary IDs do not match")
    models = model.phase_modules if model_key is None else {model_key: model}
    if set(state.get("models", {})) != set(models):
        raise ValueError("Checkpoint model keys do not match; set model_key explicitly")
    for name, module in models.items():
        module.load_state_dict(state["models"][name], strict=True)
    expected_mapping = [classifier_vocabulary.token_to_id.get(token, UNK)
                        for token in generator_vocabulary.tokens]
    if model.token_mapping.tolist() != expected_mapping:
        raise ValueError("Checkpoint token mapping does not match the vocabularies")
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Checkpoint contains non-finite model parameters")
    return {**metadata, "checkpoint_sha256": checkpoint_hash,
            "compatibility_hash": expected}


def _text_from_tokens(tokens, vocabulary: Vocabulary):
    """Remove boundaries and post-EOS padding, retaining UNK and duplicate text."""
    content, stopped = [], False
    for token in tokens:
        if type(token) is not int or not 0 <= token < len(vocabulary.tokens):
            raise ValueError("Decoder emitted an invalid token ID")
        if token == EOS:
            stopped = True
            break
        if token in (PAD, BOS):
            raise ValueError("Decoder emitted PAD/BOS before EOS")
        content.append(token)
    return " ".join(vocabulary.tokens[token] for token in content), content.count(UNK), not stopped


def export_generated_corpus(output_dir, *, checkpoint, model, run_manifest,
                            prepared_manifest, generator_vocabulary, classifier_vocabulary,
                            config, configured_epochs=80, model_key="pog"):
    """Export a new corpus independently of training; publish manifest last.

    A failed export retains its reserved directory but has no success manifest.
    Never deduplicate, change temperature, or fall back to another checkpoint.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Generation output already exists: {output}")
    if config.paper and (configured_epochs != 80
                         or prepared_manifest.get("profile") != "paper-text-preparation"):
        raise ValueError("Paper generation requires 80 epochs and paper prepared data")
    if prepared_manifest.get("tokenizer", {}).get("max_length") != config.max_length:
        raise ValueError("Generation length must match prepared data")
    provenance = load_generation_checkpoint(
        checkpoint, model=model, run_manifest=run_manifest, prepared_manifest=prepared_manifest,
        generator_vocabulary=generator_vocabulary, classifier_vocabulary=classifier_vocabulary,
        configured_epochs=configured_epochs, device=config.device, model_key=model_key)
    device = require_cuda(config.device)
    rng_seed = derived_seed(config.seed, "generation")
    rng = torch.Generator(device=device).manual_seed(rng_seed)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = {"attempts": 0, "empty_outputs": 0, "duplicate_records": 0,
                   "unk_tokens": 0, "records_with_unk": 0, "truncated_records": 0}
    records, seen = [], set()
    # Snapshot each submodule flag, including any deliberately mixed train/eval state.
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            while len(records) < config.count:
                size = min(config.batch_size, config.count - len(records))
                pending = list(range(size))
                accepted = [None] * size
                for _ in range(config.max_empty_retries + 1):
                    codes = model.generator(model.generator.sample_latent(len(pending), rng=rng))
                    if not torch.isfinite(codes).all().item():
                        raise FloatingPointError("Non-finite generated codes")
                    rollout = model.autoencoder.rollout(
                        codes, max_steps=config.max_length - 1, temperature=1.0,
                        mode="sample", generator=rng)
                    rows = rollout["tokens"].tolist()
                    if len(rows) != len(pending) or any(len(row) != config.max_length - 1 for row in rows):
                        raise ValueError("Decoder returned an invalid sampled token shape")
                    retry = []
                    for slot, tokens in zip(pending, rows):
                        diagnostics["attempts"] += 1
                        text, unks, truncated = _text_from_tokens(tokens, generator_vocabulary)
                        if not text.strip():
                            diagnostics["empty_outputs"] += 1
                            retry.append(slot)
                        else:
                            accepted[slot] = (text, unks, truncated)
                    pending = retry
                    if not pending:
                        break
                if pending:
                    raise RuntimeError(f"Empty-output retry limit exhausted after {config.max_empty_retries} retries; "
                                       f"cannot produce {config.count} records (diagnostics={diagnostics})")
                for text, unks, truncated in accepted:
                    diagnostics["duplicate_records"] += int(text in seen)
                    diagnostics["unk_tokens"] += unks
                    diagnostics["records_with_unk"] += int(unks > 0)
                    diagnostics["truncated_records"] += int(truncated)
                    seen.add(text)
                    records.append(UnlabeledRecord(f"generated:{len(records)}", text))
    finally:
        for module, training in flags:
            module.training = training
    corpus_path = output / "corpus.jsonl"
    with corpus_path.open("x", encoding="utf-8") as target:
        for row in records:
            target.write(canonical_json(asdict(row)) + "\n")
    manifest = {"format_version": 1, **provenance, "count": len(records),
                "profile": "paper-generation" if config.paper else "tiny-non-paper-generation",
                "config": asdict(config), "generation_seed": rng_seed,
                "sampling": {"mode": "sample", "temperature": 1.0, "eos_stopping": True,
                             "max_steps": config.max_length - 1, "duplicates": "retained",
                             "detokenization": "space-joined tokens; UNK retained",
                             "length_policy": "BOS plus sampled outputs; no forced EOS at cap"},
                "diagnostics": diagnostics, "corpus_sha256": file_hash(corpus_path),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(device),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    manifest["manifest_sha256"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    return load_generated_corpus(
        output, checkpoint_sha256=provenance["checkpoint_sha256"],
        data_sha256=provenance["data_sha256"], vocabulary_sha256=classifier_vocabulary.sha256,
        generator_vocabulary_sha256=generator_vocabulary.sha256,
        variant=provenance["variant"], configured_epochs=configured_epochs, paper=config.paper)


def load_generated_corpus(directory, *, checkpoint_sha256, data_sha256, vocabulary_sha256,
                          generator_vocabulary_sha256, variant, configured_epochs=80, paper=True):
    """Verify export bytes and prerequisites before final-classifier training."""
    from .classifier import GeneratedCorpus

    directory = Path(directory)
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError("Missing completed generation manifest; generate the corpus explicitly first")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    if unsigned.pop("manifest_sha256", None) != fingerprint(unsigned):
        raise ValueError("Generated manifest hash mismatch")
    expected = {"checkpoint_sha256": checkpoint_sha256, "data_sha256": data_sha256,
                "vocabulary_sha256": vocabulary_sha256,
                "generator_vocabulary_sha256": generator_vocabulary_sha256, "variant": variant,
                "epoch": configured_epochs, "epochs": configured_epochs,
                "selection": "final-configured-epoch", "format_version": 1}
    if any(not value or manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Generated checkpoint/data/vocabulary/final-epoch provenance mismatch")
    corpus = directory / "corpus.jsonl"
    if file_hash(corpus) != manifest.get("corpus_sha256"):
        raise ValueError("Generated corpus hash mismatch")
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    if (not rows or any(not isinstance(row, dict) or set(row) != {"id", "text"}
                       or not all(isinstance(v, str) and v.strip() for v in row.values()) for row in rows)
            or len({row["id"] for row in rows}) != len(rows)
            or type(manifest.get("count")) is not int or manifest["count"] != len(rows)):
        raise ValueError("Generated corpus requires unique IDs and target-free nonempty text records")
    if paper and (len(rows) != 15000 or manifest.get("profile") != "paper-generation"
                  or configured_epochs != 80):
        raise ValueError("Paper generation requires 15000 records from the final 80th epoch")
    return GeneratedCorpus(tuple(UnlabeledRecord(**row) for row in rows), manifest)
