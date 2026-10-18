"""Versioned CLINC preparation. Text operations do not import PyTorch.

See ``docs/data-protocol.md`` for artifact and provenance contracts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
PAD, UNK, BOS, EOS = range(4)
TOKEN_PATTERN = r"\w+|[^\w\s]"
PARTITIONS = ("train", "val", "test", "oos_train", "oos_val", "oos_test")
ORIGINAL_DATA = Path(__file__).resolve().parents[1] / "datasets/oos_eval/data/original"
SCHEMA_VERSION = 1


def canonical_json(value: object) -> bytes:
    """Return canonical UTF-8 JSON, including one final newline."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def content_hash(value: object) -> str:
    """Hash a JSON value with the artifact serialization rules."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hash of the exact source bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitContract:
    """Expected counts per intent and OOD split; overrides are non-paper tests."""

    labels: int = 150
    train_per_label: int = 100
    val_per_label: int = 20
    test_per_label: int = 30
    full_ood_train: int = 100
    plus_ood_train: int = 250
    ood_val: int = 100
    ood_test: int = 1000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Split contract counts must be positive integers")


@dataclass(frozen=True)
class Record:
    """Original record identity; OOD never has an intent target."""

    source: str
    partition: str
    index: int
    text: str
    is_ood: bool
    target: int | None

    @property
    def id(self) -> str:
        """Return a source/partition/index ID, not a text-derived identity."""
        return f"{self.source}:{self.partition}:{self.index}"


@dataclass(frozen=True)
class UnlabeledRecord:
    """Text-only optimization view. No target, intent, or OOD status is exposed."""

    id: str
    text: str


@dataclass(frozen=True)
class Corpus:
    """Validated original sources and the sorted training-only intent map."""

    full: dict[str, tuple[Record, ...]]
    plus: dict[str, tuple[Record, ...]]
    label_map: dict[str, int]
    source_hashes: dict[str, str]
    source_paths: dict[str, Path]
    contract: SplitContract


def _read_original(path: Path, source: str, contract: SplitContract) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing original CLINC source: {path}. Initialize the dataset submodule.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != set(PARTITIONS):
        raise ValueError(f"{source}: expected original six-partition JSON, not converted/merged supervision")
    for partition, rows in data.items():
        if not isinstance(rows, list):
            raise ValueError(f"{source}/{partition}: expected a list")
        for index, row in enumerate(rows):
            if (not isinstance(row, list) or len(row) != 2
                    or not all(isinstance(x, str) and x.strip() for x in row)):
                raise ValueError(f"{source}/{partition}/{index}: expected [nonempty text, label]")
            if (row[1].lower() in {"oos", "ood"}) != partition.startswith("oos_"):
                raise ValueError(f"{source}/{partition}/{index}: OOD must be separate, never a supervised intent")
    labels = {row[1] for row in data["train"]}
    if len(labels) != contract.labels:
        raise ValueError(f"{source}: expected {contract.labels} IND training labels, got {len(labels)}")
    for partition, count in (("train", contract.train_per_label),
                             ("val", contract.val_per_label), ("test", contract.test_per_label)):
        if Counter(row[1] for row in data[partition]) != Counter({label: count for label in labels}):
            raise ValueError(f"{source}/{partition}: wrong IND counts or label set; expected {count} per training intent")
    expected = {"oos_train": contract.full_ood_train if source == "full" else contract.plus_ood_train,
                "oos_val": contract.ood_val, "oos_test": contract.ood_test}
    for partition, count in expected.items():
        if len(data[partition]) != count:
            raise ValueError(f"{source}/{partition}: expected {count} OOD records")
    return data


def load_corpus(full_path: str | Path = ORIGINAL_DATA / "data_full.json",
                plus_path: str | Path = ORIGINAL_DATA / "data_oos_plus.json", *,
                contract: SplitContract = SplitContract()) -> Corpus:
    """Validate originals before model use; retain source IDs and source hashes.

    The plus source must have exactly the same IND and held-out OOD records as
    the full source. Repeated text alone is not treated as split leakage.
    """
    paths = {"full": Path(full_path), "plus": Path(plus_path)}
    hashes = {name: file_hash(path) for name, path in paths.items() if path.is_file()}
    raw = {name: _read_original(path, name, contract) for name, path in paths.items()}
    for partition in PARTITIONS:
        if partition != "oos_train" and raw["full"][partition] != raw["plus"][partition]:
            raise ValueError(f"plus/{partition}: must match full source record order and contents")
    label_map = {label: index for index, label in enumerate(sorted({row[1] for row in raw["full"]["train"]}))}
    converted = {}
    for source, data in raw.items():
        converted[source] = {
            partition: tuple(Record(source, partition, index, text, partition.startswith("oos_"),
                                    None if partition.startswith("oos_") else label_map[label])
                             for index, (text, label) in enumerate(rows))
            for partition, rows in data.items()}
    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise ValueError("Original sources changed while reading; preparation aborted")
    return Corpus(converted["full"], converted["plus"], label_map, hashes, paths, contract)


@dataclass(frozen=True)
class Mixture:
    """Unlabeled Dmix records and an audit manifest with no intent targets."""

    records: tuple[UnlabeledRecord, ...]
    manifest: dict


def build_dmix(corpus: Corpus, seed: int = 0, *, ind_count: int = 10000) -> Mixture:
    """Sample IND train IDs without replacement and include all plus OOD train IDs.

    ``ind_count`` is configurable only for explicit non-paper fixtures/smokes.
    Selection and the final shuffle use a private Python RNG.
    """
    if type(ind_count) is not int or not 0 < ind_count <= len(corpus.full["train"]):
        raise ValueError("Dmix IND count must be positive and no larger than IND train")
    rng = random.Random(seed)
    selected = rng.sample(list(corpus.full["train"]), ind_count) + list(corpus.plus["oos_train"])
    rng.shuffle(selected)
    if any(record.partition not in {"train", "oos_train"} for record in selected):
        raise ValueError("Dmix may contain training IDs only")
    records = tuple(UnlabeledRecord(record.id, record.text) for record in selected)
    ids = [record.id for record in records]
    manifest = {"seed": seed, "ind_count": ind_count,
                "ood_count": len(corpus.plus["oos_train"]), "ids": ids,
                "membership_sha256": content_hash(sorted(ids)),
                "ordered_ids_sha256": content_hash(ids),
                "source_hashes": corpus.source_hashes,
                "overlap": "IND subset of supervised full:train; no held-out IDs",
                "labels_hidden": True}
    return Mixture(records, manifest)


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase Unicode text; emit word runs and individual punctuation symbols."""
    return tuple(re.findall(TOKEN_PATTERN, text.lower(), flags=re.UNICODE))


@dataclass(frozen=True)
class Vocabulary:
    """Stable special IDs followed by tokens in Unicode lexical order."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens[:4] != SPECIAL_TOKENS or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary requires unique tokens and PAD/UNK/BOS/EOS at IDs 0/1/2/3")

    @cached_property
    def token_to_id(self) -> dict[str, int]:
        """Return a cached lookup; callers must not modify this derived table."""
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def sha256(self) -> str:
        """Return the hash of the ordered token list."""
        return content_hash(list(self.tokens))


def build_vocabulary(texts: Iterable[str]) -> Vocabulary:
    """Use text only; intent labels are not an input to this function."""
    tokens = {token for text in texts for token in tokenize(text)}
    return Vocabulary(SPECIAL_TOKENS + tuple(sorted(tokens - set(SPECIAL_TOKENS))))


@dataclass(frozen=True)
class Vocabularies:
    """Separate tables, token-ID mapping, and explicit text-source provenance."""

    generator: Vocabulary
    classifier: Vocabulary
    generator_to_classifier: tuple[int, ...]
    provenance: dict


def build_vocabularies(corpus: Corpus, *, profile: str = "paper") -> Vocabularies:
    """Build train-only generator and paper-transductive or train-only classifier."""
    if profile not in {"paper", "train-only"}:
        raise ValueError("Vocabulary profile must be 'paper' or 'train-only'")
    generator = build_vocabulary(record.text for record in corpus.full["train"])
    sources = [f"full:{partition}" for partition in PARTITIONS] if profile == "paper" else ["full:train", "full:oos_train"]
    sources += ["plus:oos_train"]
    rows = [record for source in sources for record in getattr(corpus, source.split(":")[0])[source.split(":")[1]]]
    classifier = build_vocabulary(record.text for record in rows)
    lookup = classifier.token_to_id
    mapping = tuple(lookup.get(token, UNK) for token in generator.tokens)
    provenance = {"profile": profile, "text_only": True,
                  "transductive_held_out_text": profile == "paper",
                  "generator_sources": ["full:train"], "classifier_sources": sources,
                  "generator_size": len(generator.tokens), "classifier_size": len(classifier.tokens)}
    return Vocabularies(generator, classifier, mapping, provenance)


@dataclass(frozen=True)
class EncodedText:
    """Padded token IDs, valid boundary-inclusive length, and truncation counts."""

    ids: tuple[int, ...]
    length: int
    original_tokens: int
    truncated_tokens: int


def encode_text(text: str, vocabulary: Vocabulary, *, max_length: int = 64) -> EncodedText:
    """Add BOS/EOS, truncate content to fit, then right-pad with PAD."""
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer >= 2, including BOS/EOS")
    tokens = tokenize(text)
    lookup = vocabulary.token_to_id
    content = tokens[:max_length - 2]
    ids = (BOS,) + tuple(lookup.get(token, UNK) for token in content) + (EOS,)
    return EncodedText(ids + (PAD,) * (max_length - len(ids)), len(ids), len(tokens), len(tokens) - len(content))


@dataclass(frozen=True)
class EmbeddingResult:
    """CUDA FP32 initial weights and JSON-compatible source/coverage metadata."""

    weights: torch.Tensor
    manifest: dict


def load_embeddings(vocabulary: Vocabulary, path: str | Path | None = None, *,
                    seed: int = 0, device: str = "cuda:0", random_only: bool = False,
                    oov_std: float = 0.1) -> EmbeddingResult:
    """Load 100d GloVe on CUDA; missing vectors use seeded N(0, oov_std²).

    Random-only initialization must be requested explicitly and is non-paper.
    The entire source is checked for dimension, finite values, and checksum.
    PAD is always zero. No download, CPU tensor, or fallback is used.
    """
    if not math.isfinite(oov_std) or oov_std <= 0:
        raise ValueError("oov_std must be positive and finite")
    if random_only and path is not None:
        raise ValueError("random_only cannot be combined with a pretrained path")
    if not random_only and (path is None or not Path(path).is_file()):
        raise FileNotFoundError("Paper preparation requires glove.6B.100d.txt. Download GloVe 6B from https://nlp.stanford.edu/projects/glove/ and supply its extracted path; use random_only=True only for non-paper smoke runs.")
    vectors = {}
    source_hash = None
    if not random_only:
        path = Path(path)
        source_hash = file_hash(path)
        wanted = set(vocabulary.tokens[4:])
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) != 101:
                    raise ValueError(f"{path}:{line_number}: expected a token and 100 dimensions, got {max(0, len(fields)-1)}")
                try:
                    values = [float(value) for value in fields[1:]]
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: invalid embedding number") from error
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{path}:{line_number}: embedding values must be finite")
                if fields[0] in wanted:
                    if fields[0] in vectors:
                        raise ValueError(f"{path}:{line_number}: duplicate vocabulary token {fields[0]!r}")
                    vectors[fields[0]] = values
        if source_hash != file_hash(path):
            raise ValueError("Embedding source changed while reading")
    import torch
    from .runtime import require_cuda

    selected_device = require_cuda(device)
    generator = torch.Generator(device=selected_device).manual_seed(seed)
    weights = torch.randn((len(vocabulary.tokens), 100), generator=generator,
                          device=selected_device, dtype=torch.float32) * oov_std
    matched = [(index, vectors[token]) for token, index in vocabulary.token_to_id.items()
               if token in vectors]
    if matched:
        indices = torch.tensor([index for index, _ in matched], device=selected_device, dtype=torch.long)
        values = torch.tensor([values for _, values in matched], device=selected_device, dtype=torch.float32)
        weights[indices] = values
    weights[PAD].zero_()
    eligible = len(vocabulary.tokens) - 4
    manifest = {"schema_version": SCHEMA_VERSION, "dimension": 100, "seed": seed,
                "profile": "random-only-smoke-non-paper" if random_only else "pretrained",
                "source": None if random_only else "GloVe 6B 100d (chosen assumption, not a paper claim)",
                "source_sha256": source_hash, "vocabulary_sha256": vocabulary.sha256,
                "matched_tokens": len(vectors), "eligible_tokens": eligible,
                "coverage": len(vectors) / eligible if eligible else 0.0,
                "oov_distribution": {"name": "normal", "mean": 0, "std": oov_std},
                "pad_id": PAD, "pad_zero": True, "dtype": "float32"}
    return EmbeddingResult(weights, manifest)


@dataclass(frozen=True)
class PreparedData:
    """In-memory preparation result; ``manifest`` is the artifact identity."""

    corpus: Corpus
    dmix: Mixture
    vocabularies: Vocabularies
    manifest: dict


def prepare_data(output_dir: str | Path, *,
                 full_path: str | Path = ORIGINAL_DATA / "data_full.json",
                 plus_path: str | Path = ORIGINAL_DATA / "data_oos_plus.json",
                 seed: int = 0, max_length: int = 64, profile: str = "paper",
                 contract: SplitContract = SplitContract(), ind_count: int = 10000,
                 embedding_manifests: dict | None = None) -> PreparedData:
    """Write stable JSON data artifacts without overwriting an existing directory.

    This text-only stage does not initialize models. A paper training consumer
    must require matching pretrained embedding manifests before training.
    Rebuild into a different fresh directory to compare identical bytes.
    """
    corpus = load_corpus(full_path, plus_path, contract=contract)
    dmix = build_dmix(corpus, seed, ind_count=ind_count)
    vocabs = build_vocabularies(corpus, profile=profile)
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    is_paper = contract == SplitContract() and ind_count == 10000 and profile == "paper" and max_length == 64
    files = {"dmix.json": [asdict(row) for row in dmix.records],
             "dmix-membership.json": dmix.manifest,
             "labels.json": corpus.label_map,
             "vocab-generator.json": list(vocabs.generator.tokens),
             "vocab-classifier.json": list(vocabs.classifier.tokens),
             "generator-to-classifier.json": list(vocabs.generator_to_classifier)}
    truncation = {}
    for partition in PARTITIONS:
        rows = corpus.full[partition]
        files[f"split-{partition}.json"] = [{"id": row.id, **asdict(row)} for row in rows]
    # Dmix encodings deliberately retain no targets or OOD status.
    groups = {f"full:{key}": rows for key, rows in corpus.full.items()}
    groups["dmix"] = dmix.records
    for name, rows in groups.items():
        for kind, vocabulary in (("generator", vocabs.generator), ("classifier", vocabs.classifier)):
            encoded = [{"id": row.id, **asdict(encode_text(row.text, vocabulary, max_length=max_length))} for row in rows]
            key = f"{name.replace(':', '-')}-{kind}"
            files[f"encoded-{key}.json"] = encoded
            truncation[key] = {"records": len(encoded),
                               "truncated_records": sum(row["truncated_tokens"] > 0 for row in encoded),
                               "truncated_tokens": sum(row["truncated_tokens"] for row in encoded)}
    for kind, metadata in (embedding_manifests or {}).items():
        if kind not in {"generator", "classifier"}:
            raise ValueError("Embedding manifest keys must be generator/classifier")
        vocabulary = getattr(vocabs, kind)
        if metadata.get("vocabulary_sha256") != vocabulary.sha256 or metadata.get("dimension") != 100:
            raise ValueError(f"{kind}: incompatible embedding manifest")
        if is_paper and metadata.get("profile") != "pretrained":
            raise ValueError("Paper data cannot declare random-only embeddings")
    manifest = {"schema_version": SCHEMA_VERSION,
                "profile": "paper-text-preparation" if is_paper else "non-paper",
                "sources": {name: {"file": path.name, "sha256": corpus.source_hashes[name]}
                            for name, path in corpus.source_paths.items()},
                "contract": asdict(contract), "seed": seed, "label_map": corpus.label_map,
                "splits": {key: {"count": len(rows), "ids_sha256": content_hash([row.id for row in rows])}
                           for key, rows in corpus.full.items()},
                "dmix": dmix.manifest, "vocabulary": vocabs.provenance,
                "vocabulary_hashes": {"generator": vocabs.generator.sha256, "classifier": vocabs.classifier.sha256},
                "tokenizer": {"pattern": TOKEN_PATTERN, "lowercase": True, "unicode": True,
                              "normalization": "none", "special_tokens": list(SPECIAL_TOKENS),
                              "max_length": max_length, "boundaries_in_length": True, "padding": "right"},
                "truncation": truncation, "embeddings": embedding_manifests or {},
                "embedding_state": "declared" if embedding_manifests else "required-before-training",
                "test_label_policy": "evaluation only; never optimization or checkpoint selection",
                "files": {name: hashlib.sha256(canonical_json(value)).hexdigest() for name, value in sorted(files.items())}}
    manifest["manifest_sha256"] = content_hash(manifest)
    if corpus.source_hashes != {name: file_hash(path) for name, path in corpus.source_paths.items()}:
        raise ValueError("Original sources changed during preparation")
    output = Path(output_dir)
    # Exclusive directory creation protects all pre-existing contents, including symlinks.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in sorted(files.items()):
        with (output / name).open("xb") as stream:
            stream.write(canonical_json(value))
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical_json(manifest))
    return PreparedData(corpus, dmix, vocabs, manifest)
