"""Artifact-driven stages. Importing this module does not start model execution."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from .config import validate_prerequisites


REPOSITORY = Path(__file__).resolve().parents[2]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _new(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}; use an explicit compatible resume")
    return path


@dataclass(frozen=True)
class PreparedArtifacts:
    directory: Path
    manifest: dict
    generator: object
    classifier: object
    mapping: tuple
    splits: dict
    dmix: object

    @property
    def sha256(self):
        return self.manifest["manifest_sha256"]


def load_prepared(directory, config):
    """Verify every prepared file before embedding or model initialization."""
    from collections import Counter
    from .data import (PARTITIONS, SCHEMA_VERSION, SPECIAL_TOKENS, TOKEN_PATTERN,
                       Mixture, Record, SplitContract, UnlabeledRecord, Vocabulary,
                       build_vocabulary, content_hash, file_hash)

    directory = Path(directory)
    if config.embedding_dim != 100:
        raise ValueError("The embedding loader requires 100d embeddings, including random-only smoke")
    manifest = _json(directory / "manifest.json")
    unsigned = dict(manifest)
    if unsigned.pop("manifest_sha256", None) != content_hash(unsigned):
        raise ValueError("Prepared manifest hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION or set(manifest["splits"]) != set(PARTITIONS):
        raise ValueError("Unsupported prepared schema or split set")
    contract = SplitContract(**manifest["contract"])
    labels = _json(directory / "labels.json")
    if (labels != manifest["label_map"] or len(labels) != config.num_classes
            or sorted(labels.values()) != list(range(config.num_classes))
            or (config.paper and contract != SplitContract())):
        raise ValueError("Prepared label map or split contract mismatch")
    tokenizer = manifest["tokenizer"]
    if (tokenizer["pattern"] != TOKEN_PATTERN or tokenizer["special_tokens"] != list(SPECIAL_TOKENS)
            or tokenizer["lowercase"] is not True or tokenizer["unicode"] is not True
            or tokenizer["boundaries_in_length"] is not True or tokenizer["padding"] != "right"):
        raise ValueError("Prepared tokenizer contract mismatch")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError(f"Prepared file hash mismatch: {name}")
    if (manifest["tokenizer"]["max_length"] != config.max_length
            or manifest["vocabulary"]["profile"] != config.vocabulary_profile
            or manifest["contract"]["labels"] != config.num_classes):
        raise ValueError("Prepared length, vocabulary profile, or intent count does not match config")
    if config.paper and manifest.get("profile") != "paper-text-preparation":
        raise ValueError("Paper training requires the full paper prepared-data profile")
    if (manifest["dmix"]["ind_count"] != config.dmix_ind_count
            or manifest["dmix"]["ood_count"] != config.dmix_ood_count):
        raise ValueError("Prepared Dmix counts do not match the resolved config")
    generator = Vocabulary(tuple(_json(directory / "vocab-generator.json")))
    classifier = Vocabulary(tuple(_json(directory / "vocab-classifier.json")))
    if manifest["vocabulary_hashes"] != {"generator": generator.sha256, "classifier": classifier.sha256}:
        raise ValueError("Prepared vocabulary hashes do not match")
    mapping = tuple(_json(directory / "generator-to-classifier.json"))
    if mapping != tuple(classifier.token_to_id.get(token, 1) for token in generator.tokens):
        raise ValueError("Prepared generator-to-classifier mapping mismatch")
    splits = {}
    for partition, metadata in manifest["splits"].items():
        rows = []
        for raw in _json(directory / f"split-{partition}.json"):
            raw = dict(raw)
            identity = raw.pop("id")
            row = Record(**raw)
            if (identity != row.id or row.source != "full" or row.partition != partition
                    or row.is_ood != partition.startswith("oos_")
                    or (row.target is not None if row.is_ood else
                        type(row.target) is not int or not 0 <= row.target < config.num_classes)):
                raise ValueError("Prepared split record identity/target mismatch")
            rows.append(row)
        ids = [row.id for row in rows]
        if (len(ids) != metadata["count"] or len(set(ids)) != len(ids)
                or content_hash(ids) != metadata["ids_sha256"]):
            raise ValueError("Prepared split membership mismatch")
        splits[partition] = tuple(rows)
    for partition, per_label in (("train", contract.train_per_label),
                                 ("val", contract.val_per_label), ("test", contract.test_per_label)):
        if Counter(row.target for row in splits[partition]) != Counter({i: per_label for i in range(contract.labels)}):
            raise ValueError("Prepared IND split counts or intent set mismatch")
    for partition, count in (("oos_train", contract.full_ood_train),
                             ("oos_val", contract.ood_val), ("oos_test", contract.ood_test)):
        if len(splits[partition]) != count:
            raise ValueError("Prepared OOD split count mismatch")
    dmix = Mixture(tuple(UnlabeledRecord(**row) for row in _json(directory / "dmix.json")),
                   _json(directory / "dmix-membership.json"))
    ids = [row.id for row in dmix.records]
    if (dmix.manifest != manifest["dmix"] or ids != dmix.manifest["ids"]
            or content_hash(sorted(ids)) != dmix.manifest["membership_sha256"]
            or content_hash(ids) != dmix.manifest["ordered_ids_sha256"]
            or len(set(ids)) != len(ids) or not dmix.manifest["labels_hidden"]
            or any(not row.id.startswith(("full:train:", "plus:oos_train:")) for row in dmix.records)):
        raise ValueError("Prepared Dmix membership mismatch")
    train_text = {row.id: row.text for row in splits["train"]}
    ind = [row for row in dmix.records if row.id.startswith("full:train:")]
    plus = [row for row in dmix.records if row.id.startswith("plus:oos_train:")]
    if (len(ind) != config.dmix_ind_count or len(plus) != contract.plus_ood_train
            or any(train_text.get(row.id) != row.text for row in ind)):
        raise ValueError("Prepared Dmix source text or count mismatch")
    vocabulary_partitions = PARTITIONS if config.vocabulary_profile == "paper" else ("train", "oos_train")
    classifier_text = [row.text for partition in vocabulary_partitions for row in splits[partition]]
    classifier_text.extend(row.text for row in plus)
    if (build_vocabulary(row.text for row in splits["train"]).sha256 != generator.sha256
            or build_vocabulary(classifier_text).sha256 != classifier.sha256):
        raise ValueError("Prepared vocabulary text-source provenance mismatch")
    return PreparedArtifacts(directory, manifest, generator, classifier, mapping, splits, dmix)


def _training_rows(prepared, config):
    # Limits are explicit non-paper settings. Do not change IDs or target mapping.
    train = prepared.splits["train"][:config.train_limit]
    val = prepared.splits["val"][:config.validation_limit]
    if not train or not val:
        raise ValueError("Training and IND validation must both be nonempty")
    if config.variant == "Mix":
        train_ids = {row.id for row in train}
        if any(row.id.startswith("full:train:") and row.id not in train_ids for row in prepared.dmix.records):
            raise ValueError("Mix training limit excludes declared Dmix IND members")
    return train, val


def _embeddings(prepared, config, seed, embedding_path, *, generator=False):
    from .data import load_embeddings
    from .state import derived_seed, fingerprint

    tables = {"classifier": prepared.classifier}
    if generator:
        tables["generator"] = prepared.generator
    results = {name: load_embeddings(vocabulary, embedding_path,
                                    seed=derived_seed(seed, "initialization"),
                                    device=config.device, random_only=config.random_only)
               for name, vocabulary in tables.items()}
    metadata = {name: result.manifest for name, result in results.items()}
    return results, metadata, fingerprint(metadata)


def _run(output, config, seed, prepared, embedding_metadata, embedding_hash, stage,
         resume, inputs=None):
    from .state import create_run, fingerprint, write_json

    values = {**config.for_seed(seed), "stage": stage, "inputs": inputs or {}}
    output = Path(output)
    expected = fingerprint({"config": values, "data_hash": prepared.sha256,
                            "embedding_hash": embedding_hash})
    if resume is None:
        _new(output)
        manifest = create_run(output, values, prepared.sha256, embedding_hash,
                              REPOSITORY, config.device)
        write_json(output / "embeddings.json", embedding_metadata)
    else:
        manifest = _json(output / "manifest.json")
        if (manifest.get("compatibility_hash") != expected or manifest.get("config") != values
                or manifest.get("data_hash") != prepared.sha256 or manifest.get("embedding_hash") != embedding_hash
                or _json(output / "config.json") != values
                or _json(output / "embeddings.json") != embedding_metadata):
            raise ValueError("Resume data, embeddings, config, stage, or entropy inputs mismatch")
        candidates = sorted((output / "checkpoints").glob("boundary-*.pt"))
        if not candidates:
            raise ValueError("Run has no durable completed boundary; preserve this run and use a new output directory")
        if (Path(resume).resolve() != candidates[-1].resolve()
                or Path(resume).parent.resolve() != (output / "checkpoints").resolve()):
            raise ValueError("Resume must name the latest completed boundary in this run")
    return manifest


def _publish(path, value):
    """An interrupted final publication may be repeated only with identical data."""
    from .state import write_json

    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Existing completion artifact differs: {path}")
    else:
        write_json(path, value)


class BoundaryBatches:
    """A CUDA shuffle stream with a saved permutation and next-record position."""

    def __init__(self, size, batch_size, generator):
        if size < 1 or batch_size < 1:
            raise ValueError("Loader size and batch size must be positive")
        self.size, self.batch_size, self.generator = size, batch_size, generator
        self.order, self.position, self.restarts = [], 0, 0

    def take(self):
        import torch

        if self.position == len(self.order):
            self.order = torch.randperm(self.size, device=self.generator.device,
                                        generator=self.generator).tolist()
            self.position = 0
            self.restarts += 1
        result = self.order[self.position:self.position + self.batch_size]
        self.position += len(result)
        return result

    def reset(self):
        self.order, self.position = [], 0

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "position": self.position, "restarts": self.restarts}

    def load_state_dict(self, value):
        order, position = value["order"], value["position"]
        if (value["size"] != self.size or value["batch_size"] != self.batch_size
                or (order and sorted(order) != list(range(self.size)))
                or not 0 <= position <= len(order) or value["restarts"] < 0):
            raise ValueError("Incompatible loader permutation or position")
        self.order, self.position, self.restarts = list(order), position, value["restarts"]


def generator_phases(config):
    """Ordered separate optimizer steps, including every configured repeat."""
    phases = ["reconstruction", "auxiliary"]
    phases += [f"critic:{i}" for i in range(config.critic_iterations)]
    phases += ["generator_wasserstein", "generator_entropy"]
    if config.mode == "aepog":
        phases += [f"dmix_reconstruction:{i}" for i in range(config.dmix_reconstruction_iterations)]
        phases += [f"dmix_critic:{i}" for i in range(config.dmix_critic_iterations)]
    return tuple(phases)


def _build_generator(prepared, config, embeddings=None):
    from pseudo_ood_generation.builders import build_model

    tables = {} if embeddings is None else {
        "generator_embeddings": embeddings["generator"].weights,
        "auxiliary_embeddings": embeddings["classifier"].weights}
    return build_model(len(prepared.generator.tokens), len(prepared.classifier.tokens),
                       prepared.mapping, **config.model_kwargs(), **tables)


def _encode(rows, vocabulary, config):
    import torch
    from .data import encode_text

    return torch.tensor([encode_text(row.text, vocabulary, max_length=config.max_length).ids
                         for row in rows], device=config.device, dtype=torch.long)


def train_generator(*, prepared, output, config, seed, embedding_path=None, resume=None,
                    stop_after_phases=None, checkpoint_interval_seconds=None,
                    retain_checkpoints=False, on_checkpoint=None):
    """Train at phase boundaries. See CheckpointPolicy for the synchronous callback.

    None saves every phase. A positive interval also saves at epoch completion,
    an explicit stop, and completion. Retention keeps the newest resume state.
    These operational options do not change experiment compatibility.
    """
    import torch
    from pseudo_ood_generation.trainer import POGSettings, PseudoOODGenerationTrainer
    from .generation import checkpoint_generation_metadata
    from .state import Boundary, derived_seed, log_phase, restore_checkpoint, save_checkpoint, seed_run
    from .checkpoint_policy import CheckpointPolicy, model_checkpoint_estimate, reconcile_logs

    policy = CheckpointPolicy(output, checkpoint_interval_seconds, retain_checkpoints, on_checkpoint)
    validate_prerequisites(config, "train-generator", embedding_path=embedding_path)
    if stop_after_phases is not None and (type(stop_after_phases) is not int or stop_after_phases < 1):
        raise ValueError("stop_after_phases must be a positive integer")
    config.for_seed(seed)
    prepared = load_prepared(prepared, config)
    train, _ = _training_rows(prepared, config)
    if resume is None:
        _new(output)
    streams = seed_run(seed, config.device)
    streams["training_noise"] = torch.Generator(device=config.device).manual_seed(derived_seed(seed + 2, "initialization"))
    embeddings, metadata, embedding_hash = _embeddings(prepared, config, seed, embedding_path, generator=True)
    manifest = _run(output, config, seed, prepared, metadata, embedding_hash,
                    "train-generator", resume)
    # Model initialization is independent of the loader/noise streams.
    torch.manual_seed(derived_seed(seed, "initialization"))
    model = _build_generator(prepared, config, embeddings)
    trainer = PseudoOODGenerationTrainer(model, POGSettings(**config.pog_settings()))
    output = Path(output)
    tokens = _encode(train, prepared.generator, config)
    labels = torch.tensor([row.target for row in train], device=config.device, dtype=torch.long)
    dmix_tokens = _encode(prepared.dmix.records, prepared.generator, config) if config.mode == "aepog" else None
    loaders = {"ind": BoundaryBatches(len(train), config.batch_size, streams["loader"])}
    if dmix_tokens is not None:
        # A separate stream, not an interleaved use of the IND shuffle stream.
        streams["dmix_loader"] = torch.Generator(device=config.device).manual_seed(derived_seed(seed + 1, "loader"))
        loaders["dmix"] = BoundaryBatches(len(dmix_tokens), config.batch_size, streams["dmix_loader"])
    phases = generator_phases(config)
    epoch = batch = phase_index = serial = 0
    current = {}
    checkpoint = None
    arguments = dict(models={"pog": model}, optimizers=trainer.optimizers, generators=streams,
                     compatibility_hash=manifest["compatibility_hash"])
    if resume is not None:
        restored = restore_checkpoint(resume, **arguments, device=config.device)
        saved = restored["extra"]
        trainer.load_state_dict(saved["trainer"])
        epoch, batch, phase_index, serial = (saved[name] for name in ("epoch", "batch", "phase_index", "serial"))
        current = saved["current_batches"]
        batches_per_epoch = math.ceil(len(train) / config.batch_size)
        if (any(type(value) is not int for value in (epoch, batch, phase_index, serial))
                or not 0 <= epoch <= config.generator_epochs
                or not 0 <= batch < batches_per_epoch or not 0 <= phase_index < len(phases)
                or serial != (epoch * batches_per_epoch + batch) * len(phases) + phase_index
                or (epoch == config.generator_epochs and (batch != 0 or phase_index != 0))
                or bool(current) != (phase_index != 0)
                or (current and set(current) != set(loaders))):
            raise ValueError("Incompatible generator phase/epoch/loader position")
        if (restored["boundary"] != Boundary(epoch, batch, "complete" if epoch == config.generator_epochs else phases[phase_index])
                or restored["temperature"] != trainer.temperature
                or restored["policy_baseline"] != trainer.baseline):
            raise ValueError("Generator boundary, schedule, or policy baseline mismatch")
        for name, loader in loaders.items():
            loader.load_state_dict(saved["loaders"][name])
            if current:
                ids = current[name]
                if (not ids or any(type(index) is not int or not 0 <= index < loader.size for index in ids)
                        or ids != loader.order[max(0, loader.position - len(ids)):loader.position]):
                    raise ValueError("Saved current batch does not match its loader cursor")
        checkpoint = output / "checkpoints" / Path(resume).name
        reconcile_logs(output, "generator", serial, saved.get("phase_log"))
    completed_here = 0
    while epoch < config.generator_epochs:
        trainer.set_epoch(epoch)
        if not current:
            current = {name: loader.take() for name, loader in loaders.items()}
        ind = tokens[current["ind"]]
        phase = phases[phase_index].split(":")[0]
        rng = streams["training_noise"]
        torch.cuda.synchronize(ind.device)
        phase_started = time.perf_counter()
        if phase == "reconstruction":
            record = trainer.reconstruction_step(ind, rng=rng)
        elif phase == "auxiliary":
            record = trainer.auxiliary_step(ind, labels[current["ind"]], rng=rng)
        elif phase == "critic":
            record = trainer.critic_step(ind, rng=rng)
        elif phase == "generator_wasserstein":
            record = trainer.generator_wasserstein_step(len(ind), rng=rng)
        elif phase == "generator_entropy":
            record = trainer.generator_entropy_step(len(ind), rng=rng)
        elif phase == "dmix_reconstruction":
            record = trainer.reconstruction_step(dmix_tokens[current["dmix"]], dmix=True, rng=rng)
        elif phase == "dmix_critic":
            record = trainer.critic_step(dmix_tokens[current["dmix"]], dmix=True, rng=rng)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        torch.cuda.synchronize(ind.device)
        record["seconds"] = time.perf_counter() - phase_started
        phase_index += 1
        generation = None
        if phase_index == len(phases):
            phase_index = 0
            current = {}
            batch += 1
            if batch == math.ceil(len(train) / config.batch_size):
                epoch += 1
                batch = 0
                loaders["ind"].reset()
                generation = checkpoint_generation_metadata(
                    mode=config.mode, completed_epochs=epoch, configured_epochs=config.generator_epochs,
                    data_sha256=prepared.sha256, generator_vocabulary=prepared.generator,
                    classifier_vocabulary=prepared.classifier)
        serial += 1
        completed_here += 1
        stopping = stop_after_phases is not None and completed_here >= stop_after_phases
        save_now = policy.due(force=generation is not None or stopping or epoch == config.generator_epochs)
        next_checkpoint = output / "checkpoints" / f"boundary-{serial:09d}.pt"
        phase_log = {**record, "boundary_serial": serial,
                     "checkpoint": next_checkpoint.name if save_now else None}
        if not save_now:
            log_phase(output, phase_log)
            continue
        extra = {"trainer": trainer.state_dict(), "epoch": epoch, "batch": batch,
                  "phase_index": phase_index, "serial": serial, "current_batches": current,
                  "loaders": {name: loader.state_dict() for name, loader in loaders.items()},
                  "phase_log": phase_log}
        if generation is not None:
            extra["generation"] = generation
        policy.guard(checkpoint, estimate=model_checkpoint_estimate(arguments["models"]))
        checkpoint_started = time.perf_counter()
        save_checkpoint(next_checkpoint, **arguments,
                        boundary=Boundary(epoch, batch, "complete" if epoch == config.generator_epochs else phases[phase_index]),
                        temperature=trainer.temperature, policy_baseline=trainer.baseline, extra=extra)
        checkpoint = next_checkpoint
        phase_log["checkpoint_io_seconds"] = time.perf_counter() - checkpoint_started
        log_phase(output, phase_log)
        policy.publish(checkpoint, final=checkpoint if epoch == config.generator_epochs else None)
        if epoch != config.generator_epochs:
            policy.notify_and_prune(checkpoint)
        if stopping:
            break
    result = {"complete": epoch == config.generator_epochs, "checkpoint": str(checkpoint),
              "completed_epochs": epoch, "phase_boundaries": serial,
              "update_counts": trainer.update_counts, "sample_counts": trainer.sample_counts}
    if result["complete"]:
        _publish(output / "summary.json", result)
        policy.publish(checkpoint, final=checkpoint)
        policy.notify_and_prune(checkpoint)
    return result


def _generator_run(directory, config, seed, prepared):
    from .state import fingerprint

    directory = Path(directory)
    manifest = _json(directory / "manifest.json")
    expected_config = {**config.for_seed(seed), "stage": "train-generator", "inputs": {}}
    if (manifest["config"] != expected_config or manifest["data_hash"] != prepared.sha256
            or manifest["compatibility_hash"] != fingerprint({key: manifest[key]
                for key in ("config", "data_hash", "embedding_hash")})
            or manifest["embedding_hash"] != fingerprint(_json(directory / "embeddings.json"))):
        raise ValueError("Generator run config, seed, or prepared data mismatch")
    summary = _json(directory / "summary.json")
    checkpoint = Path(summary["checkpoint"])
    # Resolve the basename within this run, so a copied run remains usable.
    checkpoint = directory / "checkpoints" / checkpoint.name
    if not summary["complete"] or summary["completed_epochs"] != config.generator_epochs or not checkpoint.is_file():
        raise ValueError("Generator must finish its final configured epoch before export")
    return manifest, checkpoint


def generate(*, prepared, generator_run, output, config, seed):
    """Export only from the final completed configured generator epoch."""
    from .generation import GenerationConfig, export_generated_corpus

    validate_prerequisites(config, "generate")
    if config.generation_temperature != 1.0:
        raise ValueError("The recorded generation protocol requires sampling temperature 1.0")
    _new(output)
    prepared = load_prepared(prepared, config)
    manifest, checkpoint = _generator_run(generator_run, config, seed, prepared)
    model = _build_generator(prepared, config)
    corpus = export_generated_corpus(
        output, checkpoint=checkpoint, model=model, run_manifest=manifest,
        prepared_manifest=prepared.manifest, generator_vocabulary=prepared.generator,
        classifier_vocabulary=prepared.classifier, configured_epochs=config.generator_epochs,
        config=GenerationConfig(seed=seed, paper=config.paper, count=config.generated_count,
                                batch_size=config.batch_size, max_length=config.max_length,
                                max_empty_retries=config.empty_retry_limit, device=config.device))
    from .state import write_json
    write_json(Path(output) / "config.json", config.for_seed(seed))
    write_json(Path(output) / "generator-run-manifest.json", manifest)
    return {"output": str(output), "manifest": corpus.manifest}


def generated_token_ids(text, prepared, max_length):
    """Invert the export's space join, then map saved generator IDs to CNN IDs.

    Tokenizer tokens contain no whitespace. This is lossless for retained content,
    including literal <unk>. A capped rollout has no forced EOS or lost last token.
    """
    from .data import BOS, EOS, PAD

    words = text.split(" ")
    lookup = prepared.generator.token_to_id
    if (not words or any(not word or word not in lookup or lookup[word] in (PAD, BOS, EOS) for word in words)
            or len(words) > max_length - 1):
        raise ValueError("Generated text is not an exact space-joined generator token sequence")
    ids = [BOS] + [prepared.mapping[lookup[word]] for word in words]
    if len(ids) < max_length:
        ids.append(EOS)
    return tuple(ids + [PAD] * (max_length - len(ids)))


def _generated(prepared, config, seed, generated, generator_run, embedding_path):
    from .generation import MODES, load_generated_corpus
    from .state import file_hash

    if generated is None or generator_run is None:
        raise ValueError("Generated variants require --generated and --generator-run; run both explicit stages first")
    _, checkpoint = _generator_run(generator_run, config, seed, prepared)
    corpus = load_generated_corpus(
        generated, checkpoint_sha256=file_hash(checkpoint), data_sha256=prepared.sha256,
        vocabulary_sha256=prepared.classifier.sha256, generator_vocabulary_sha256=prepared.generator.sha256,
        variant=MODES[config.mode], configured_epochs=config.generator_epochs, paper=config.paper)
    # The exporter labels mechanism/profile, while the config labels the CNN family.
    # Check the original hashed artifact first; adapt only the in-memory consumer view.
    expected_profile = "paper-generation" if config.paper else "tiny-non-paper-generation"
    if corpus.manifest["profile"] != expected_profile or corpus.manifest["mode"] != config.mode:
        raise ValueError("Generated profile or mechanism mismatch")
    adapted = {**corpus.manifest, "variant": config.variant, "profile": config.profile}
    adapted["export_manifest_sha256"] = adapted.pop("manifest_sha256")
    adapted["view"] = "validated-classifier-consumer; original export manifest retained separately"
    validate_prerequisites(config, "train-classifier", embedding_path=embedding_path,
                           generated_manifest={**corpus.manifest, "variant": config.variant},
                           generated_records=[asdict(row) for row in corpus.records],
                           data_sha256=prepared.sha256, vocabulary_sha256=prepared.classifier.sha256,
                           checkpoint_sha256=file_hash(checkpoint))
    from .classifier import GeneratedCorpus
    return GeneratedCorpus(corpus.records, adapted), corpus.manifest


class _PauseTraining(Exception):
    pass


def train_classifier(*, prepared, output, config, seed, embedding_path=None,
                     generated=None, generator_run=None, resume=None, stop_after_epochs=None,
                     checkpoint_interval_seconds=None, retain_checkpoints=False, on_checkpoint=None):
    """Train a CNN. Every epoch is a mandatory completed boundary.

    The interval cannot skip epoch or selection saves. Optional retention keeps
    latest, best-selected, and final files. See CheckpointPolicy for the callback.
    """
    import torch
    from .classifier import ClassifierConfig, FinalClassifierTrainer, make_entropy_source
    from .state import Boundary, log_phase, restore_checkpoint, save_checkpoint, seed_run
    from .checkpoint_policy import CheckpointPolicy, model_checkpoint_estimate, reconcile_logs

    policy = CheckpointPolicy(output, checkpoint_interval_seconds, retain_checkpoints, on_checkpoint)
    # Validate missing embeddings before any CUDA allocation or model construction.
    if config.paper and (embedding_path is None or not Path(embedding_path).is_file()):
        raise FileNotFoundError("Paper training requires a supplied glove.6B.100d.txt file")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    config.for_seed(seed)
    prepared = load_prepared(prepared, config)
    train, validation = _training_rows(prepared, config)
    corpus = original_generated = None
    if config.uses_generator:
        corpus, original_generated = _generated(prepared, config, seed, generated, generator_run, embedding_path)
    else:
        if generator_run is not None:
            raise ValueError("This classifier variant forbids --generator-run")
        validate_prerequisites(config, "train-classifier", embedding_path=embedding_path,
                               generated_manifest={} if generated is not None else None)
    settings = ClassifierConfig(**config.classifier_settings(seed))
    source = make_entropy_source(settings, prepared.classifier, train=train,
                                 dmix=prepared.dmix if config.variant == "Mix" else None,
                                 generated=corpus, data_sha256=prepared.sha256,
                                 checkpoint_sha256=None if corpus is None else corpus.manifest["checkpoint_sha256"])
    mapped = {} if corpus is None else {row.id: generated_token_ids(row.text, prepared, config.max_length)
                                       for row in corpus.records}
    if resume is None:
        _new(output)
    streams = seed_run(seed, config.device)
    embeddings, metadata, embedding_hash = _embeddings(prepared, config, seed, embedding_path)
    inputs = {"entropy": None if source is None else source.manifest,
              "generated_export": original_generated,
              "generated_encoding": None if corpus is None else "inverse-space-join; saved generator-to-classifier IDs; no forced EOS at cap"}
    manifest = _run(output, config, seed, prepared, metadata, embedding_hash,
                    "train-classifier", resume, inputs)

    class ArtifactClassifierTrainer(FinalClassifierTrainer):
        def _encode(self, records):
            if mapped and all(row.id in mapped for row in records):
                return torch.tensor([mapped[row.id] for row in records], device=self.device, dtype=torch.long)
            if any(row.id in mapped for row in records):
                raise ValueError("Generated and original records cannot share an encoding batch")
            return super()._encode(records)

    trainer = ArtifactClassifierTrainer(settings, prepared.classifier, embeddings["classifier"])
    streams.update({f"classifier_{name}": rng for name, rng in trainer.generators.items()})
    output = Path(output)
    arguments = dict(models={"classifier": trainer.model}, optimizers={"classifier": trainer.optimizer},
                     generators=streams, compatibility_hash=manifest["compatibility_hash"])
    checkpoint = None
    if resume is not None:
        restored = restore_checkpoint(resume, **arguments, device=config.device)
        trainer.restore_selection(restored["extra"]["selection"])
        if (restored["boundary"] != Boundary(trainer.completed_epoch, 0, "complete" if trainer.completed_epoch == config.classifier_epochs else "classifier_epoch")
                or restored["temperature"] != 1.0 or restored["policy_baseline"] != 0.0):
            raise ValueError("Classifier selection and checkpoint boundary disagree")
        checkpoint = output / "checkpoints" / Path(resume).name
        reconcile_logs(output, "classifier", trainer.completed_epoch, restored["extra"].get("phase_log"))
    completed_here = 0

    def on_epoch(event, current):
        nonlocal checkpoint, completed_here
        policy.guard(checkpoint, estimate=model_checkpoint_estimate(arguments["models"]))
        checkpoint = output / "checkpoints" / f"boundary-{event.epoch:09d}.pt"
        phase_log = {"phase": "classifier_epoch", **event.metrics, "selected": event.selected}
        checkpoint_started = time.perf_counter()
        save_checkpoint(checkpoint, **arguments,
                        boundary=Boundary(event.epoch, 0, "complete" if event.final else "classifier_epoch"),
                        temperature=1.0, policy_baseline=0.0,
                         extra={"selection": current.selection_state(), "metrics": event.metrics,
                                "phase_log": phase_log,
                                 "loader_boundary": "epoch-complete; next epoch creates fresh independent permutations"})
        phase_log["checkpoint_io_seconds"] = time.perf_counter() - checkpoint_started
        log_phase(output, phase_log)
        policy.publish(checkpoint, selected=output / "checkpoints" / f"boundary-{current.selected_epoch:09d}.pt",
                       final=checkpoint if event.final else None)
        if not event.final:
            policy.notify_and_prune(checkpoint)
        completed_here += 1
        if stop_after_epochs is not None and completed_here >= stop_after_epochs and not event.final:
            raise _PauseTraining

    try:
        trainer.fit(train, validation, source=source, on_epoch=on_epoch)
    except _PauseTraining:
        pass
    result = {"complete": trainer.completed_epoch == config.classifier_epochs,
              "completed_epochs": trainer.completed_epoch, "checkpoint": str(checkpoint),
              "selected_epoch": trainer.selected_epoch,
              "selected_checkpoint": f"checkpoints/boundary-{trainer.selected_epoch:09d}.pt",
              "final_checkpoint": f"checkpoints/boundary-{config.classifier_epochs:09d}.pt",
              "best_validation_accuracy": trainer.best_validation_accuracy,
              "selection": "best IND-validation accuracy; earliest epoch on ties"}
    if result["complete"]:
        _publish(output / "summary.json", result)
        policy.publish(checkpoint, selected=output / result["selected_checkpoint"], final=checkpoint)
        policy.notify_and_prune(checkpoint)
    return result


def evaluate(*, prepared, classifier_run, output, config, seed, calibrate=False):
    """Freeze selected epoch identity before reading test labels for prediction."""
    import torch
    from .classifier import ClassifierConfig, build_final_classifier, predict_confidence
    from .evaluation import EvaluationProvenance, calibrate_threshold, prediction_records, save_evaluation
    from .runtime import require_cuda
    from .state import file_hash, fingerprint

    _new(output)
    prepared = load_prepared(prepared, config)
    run = Path(classifier_run)
    manifest, summary = _json(run / "manifest.json"), _json(run / "summary.json")
    values = manifest["config"]
    expected_values = {**config.for_seed(seed), "stage": "train-classifier", "inputs": values.get("inputs")}
    embedding_metadata = _json(run / "embeddings.json")
    if (values != expected_values or manifest["data_hash"] != prepared.sha256
            or manifest["compatibility_hash"] != fingerprint({key: manifest[key]
                for key in ("config", "data_hash", "embedding_hash")})
            or manifest["embedding_hash"] != fingerprint(embedding_metadata)
            or not summary["complete"] or summary["completed_epochs"] != config.classifier_epochs):
        raise ValueError("Classifier config/data mismatch or incomplete final training")
    selected = summary["selected_epoch"]
    expected_path = f"checkpoints/boundary-{selected:09d}.pt"
    if summary["selected_checkpoint"] != expected_path or not 1 <= selected <= config.classifier_epochs:
        raise ValueError("Invalid selected checkpoint identity")
    checkpoint = run / expected_path
    checkpoint_hash = file_hash(checkpoint)
    provenance = EvaluationProvenance(seed, config.mode if config.mode.startswith("pog-no-") else config.variant,
                                      prepared.sha256, prepared.classifier.sha256,
                                      expected_path, checkpoint_hash)
    device = require_cuda(config.device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if (file_hash(checkpoint) != checkpoint_hash or state["compatibility_hash"] != manifest["compatibility_hash"]
            or state["boundary"]["epoch"] != selected
            or state["extra"]["selection"]["selected_epoch"] != selected):
        raise ValueError("Selected checkpoint provenance mismatch")
    final_path = run / f"checkpoints/boundary-{config.classifier_epochs:09d}.pt"
    if summary["final_checkpoint"] != str(final_path.relative_to(run)):
        raise ValueError("Final checkpoint identity mismatch")
    final_state = torch.load(final_path, map_location=device, weights_only=False)
    final_selection = final_state["extra"]["selection"]
    if (final_state["compatibility_hash"] != manifest["compatibility_hash"]
            or final_state["boundary"] != {"epoch": config.classifier_epochs, "batch": 0, "next_phase": "complete"}
            or final_selection["completed_epoch"] != config.classifier_epochs
            or final_selection["selected_epoch"] != selected
            or final_selection["best_validation_accuracy"] != summary["best_validation_accuracy"]
            or set(final_selection["selected_state"]) != set(state["models"]["classifier"])
            or any(not torch.equal(value, final_selection["selected_state"][key])
                   for key, value in state["models"]["classifier"].items())):
        raise ValueError("Selected checkpoint is not the final validation-selected CNN")
    # Inference needs trained weights, not the original large embedding file.
    from dataclasses import replace
    settings = replace(ClassifierConfig(**config.classifier_settings(seed)), paper=False)
    model = build_final_classifier(settings, prepared.classifier)
    model.load_state_dict(state["models"]["classifier"], strict=True)

    def predictions(split):
        rows = prepared.splits[split] + prepared.splits[f"oos_{split}"]
        probabilities = []
        for start in range(0, len(rows), config.batch_size):
            batch = _encode(rows[start:start + config.batch_size], prepared.classifier, config)
            probabilities.extend(predict_confidence(model, batch)["logits"].softmax(-1).tolist())
        return prediction_records([{"id": row.id, **asdict(row)} for row in rows], probabilities,
                                  provenance=provenance, split=split)

    threshold = calibrate_threshold(predictions("val")) if calibrate else None
    deviations = [] if config.paper else ["Tiny random-only non-paper run; not paper acceptance or numerical reproduction"]
    comparable_config = config.to_dict()
    comparable_config.pop("seeds")
    comparable_embeddings = {name: {key: value for key, value in metadata.items() if key != "seed"}
                             for name, metadata in embedding_metadata.items()}
    assumptions = [
        f"Resolved experiment settings excluding seed list: SHA256={fingerprint(comparable_config)}",
        f"Embedding source metadata excluding initialization seed: SHA256={fingerprint(comparable_embeddings)}",
        f"Source revisions and dirty-state flags: SHA256={fingerprint(manifest['source'])}",
        f"Execution runtime identity: SHA256={fingerprint(manifest['runtime'])}",
    ]
    report = save_evaluation(output, predictions("test"), deployment_threshold=threshold,
                             deviations=deviations, assumptions=assumptions)
    from .state import write_json
    write_json(Path(output) / "config.json", config.for_seed(seed))
    write_json(Path(output) / "run-manifest.json", manifest)
    return report


def aggregate(*, reports, output, expected_seeds=tuple(range(10))):
    """Aggregate explicit summary files only; never discover or skip missing runs."""
    from .evaluation import aggregate_reports
    from .state import write_json

    _new(output)
    result = aggregate_reports([_json(path) for path in reports], expected_seeds=expected_seeds)
    write_json(output, result)
    return result


def smoke(*, output, device="cuda:0"):
    """Bounded synthetic non-paper flow for all modes, variants, and GP phases."""
    import torch
    import platform
    from .config import smoke_config
    from .data import SplitContract, prepare_data
    from .runtime import require_cuda
    from .state import derived_seed, write_json

    output = _new(output)
    selected = require_cuda(device)
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize(selected)
    torch.cuda.reset_peak_memory_stats(selected)
    started = time.perf_counter()
    contract = SplitContract(labels=2, train_per_label=4, val_per_label=2, test_per_label=2,
                             full_ood_train=2, plus_ood_train=2, ood_val=2, ood_test=2)
    fixture = {}
    for partition, count in (("train", 4), ("val", 2), ("test", 2)):
        fixture[partition] = [[f"{label} request {partition} number {i}", label]
                              for label in ("book", "cancel") for i in range(count)]
        fixture[f"oos_{partition}"] = [[f"outside topic {partition} {i}", "oos"] for i in range(2)]
    full, plus = output / "data_full.json", output / "data_oos_plus.json"
    write_json(full, fixture)
    write_json(plus, fixture)
    prepared = output / "prepared"
    prepare_data(prepared, full_path=full, plus_path=plus, contract=contract,
                 seed=derived_seed(0, "prepare"), max_length=8, ind_count=4)
    reports = []
    cases = (("MSP", "pog"), ("Perturb", "pog"), ("Mix", "pog"),
             ("POG", "pog"), ("AEPOG", "aepog"),
             ("POG", "pog-no-noise"), ("POG", "pog-no-soft-token"))
    for variant, mode in cases:
        config = smoke_config(variant, mode=mode, device=device)
        name = mode if variant in {"POG", "AEPOG"} else variant
        case = output / name
        case.mkdir()
        gen_run, corpus = None, None
        if config.uses_generator:
            gen_run, corpus = case / "generator", case / "generated"
            train_generator(prepared=prepared, output=gen_run, config=config, seed=0)
            generate(prepared=prepared, generator_run=gen_run, output=corpus, config=config, seed=0)
        train_classifier(prepared=prepared, output=case / "classifier", config=config, seed=0,
                         generated=corpus, generator_run=gen_run)
        report = evaluate(prepared=prepared, classifier_run=case / "classifier", output=case / "evaluation",
                          config=config, seed=0, calibrate=True)
        aggregate(reports=[case / "evaluation/summary.json"], output=case / "aggregate.json", expected_seeds=(0,))
        reports.append({"variant": variant, "mode": mode, "metrics": report["metrics"]})
        del config
    torch.cuda.synchronize(selected)
    result = {"profile": "tiny-smoke-non-paper", "random_only": True, "cases": reports,
              "seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected),
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(selected)},
               "acceptance": "tiny functional verification; not paper-score acceptance"}
    write_json(output / "summary.json", result)
    return result
