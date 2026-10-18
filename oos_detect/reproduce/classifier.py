"""Independent CUDA final-classifier training; no CLI or artifact writes."""

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch
from torch.nn import functional as F

from pseudo_ood_generation.components.auxilliary_classifier import SentenceCNN

from .data import (
    EmbeddingResult, Mixture, Record, UnlabeledRecord, Vocabulary,
    content_hash, encode_text,
)
from .runtime import require_cuda
from .state import derived_seed


VARIANTS = ("MSP", "Perturb", "Mix", "POG", "AEPOG")


@dataclass(frozen=True)
class ClassifierConfig:
    """Paper dimensions by default; small overrides require paper=False."""

    variant: str = "MSP"
    paper: bool = True
    epochs: int = 30
    generator_epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    alpha: float = 1.0
    perturb_probability: float = 0.5
    seed: int = 0
    embedding_dim: int = 100
    num_classes: int = 150
    out_channels: int = 128
    kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    mlp_hidden_size: int = 512
    max_length: int = 64
    device: str = "cuda:0"

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        dimensions = (self.epochs, self.generator_epochs, self.batch_size,
                      self.embedding_dim, self.num_classes, self.out_channels,
                      self.mlp_hidden_size, self.max_length)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Classifier counts and dimensions must be positive integers")
        if self.num_classes < 2 or self.max_length < 2:
            raise ValueError("At least two intents and two boundary tokens are required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1)")
        if not self.kernel_sizes or any(type(k) is not int or k <= 0 for k in self.kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not math.isfinite(self.perturb_probability) or not 0 <= self.perturb_probability <= 1:
            raise ValueError("perturb_probability must be in [0, 1]")
        if torch.device(self.device).type != "cuda":
            raise ValueError("CUDA is required; CPU fallback is disabled")
        if self.paper and (
            self.epochs != 30 or self.generator_epochs != 80 or self.alpha != 1
            or self.perturb_probability != 0.5 or self.embedding_dim != 100
            or self.num_classes != 150 or self.out_channels != 128
            or tuple(self.kernel_sizes) != (2, 3, 4, 5)
            or self.mlp_hidden_size != 512 or self.max_length != 64
        ):
            raise ValueError("Paper settings require 80/30 epochs, alpha=1, p=0.5 and the full CNN dimensions")


def build_final_classifier(config: ClassifierConfig, vocabulary: Vocabulary,
                           embeddings: EmbeddingResult | None = None) -> SentenceCNN:
    """Initialize fresh CNN parameters; copy only original embedding values."""
    if config.paper and (embeddings is None
                         or embeddings.manifest.get("profile") != "pretrained"
                         or not embeddings.manifest.get("source_sha256")):
        raise ValueError("Paper classifier requires pretrained embeddings from load_embeddings; no random substitution")
    if embeddings is not None:
        if (embeddings.manifest.get("vocabulary_sha256") != vocabulary.sha256
                or embeddings.manifest.get("dimension") != config.embedding_dim):
            raise ValueError("Embedding vocabulary or dimension does not match classifier")
    device = require_cuda(config.device)
    if embeddings is not None:
        if (embeddings.weights.device != device
                or not torch.isfinite(embeddings.weights).all()):
            raise ValueError("Embedding weights must be finite and on the selected CUDA device")
    # Fork only the selected GPU's RNG; auxiliary initialization/state is untouched.
    with torch.random.fork_rng(devices=[device.index]), torch.cuda.device(device):
        torch.cuda.manual_seed(derived_seed(config.seed, "initialization"))
        return SentenceCNN(
            len(vocabulary.tokens), embedding_dim=config.embedding_dim,
            num_classes=config.num_classes, out_channels=config.out_channels,
            kernel_sizes=config.kernel_sizes, mlp_hidden_size=config.mlp_hidden_size,
            embeddings=None if embeddings is None else embeddings.weights, device=device,
        )


def _logits(logits):
    if (logits.device.type != "cuda" or logits.ndim != 2
            or min(logits.shape) < 1 or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite, nonempty CUDA logits [batch, intents]")


def negative_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean sum(p log p); minimizing this value favors uniform predictions."""
    _logits(logits)
    log_probability = logits.log_softmax(-1)
    return (log_probability.exp() * log_probability).sum(-1).mean()


def classifier_objective(ind_logits, targets, entropy_logits=None, *, alpha=1.0):
    """CE labels are accepted only for the IND logits, never entropy inputs."""
    _logits(ind_logits)
    if (targets.device != ind_logits.device or targets.dtype != torch.long
            or targets.shape != ind_logits.shape[:1]
            or ((targets < 0) | (targets >= ind_logits.shape[-1])).any()):
        raise ValueError("Expected CUDA IND targets within the intent label range")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    ce = F.cross_entropy(ind_logits, targets)
    entropy = ind_logits.new_zeros(())
    if entropy_logits is not None:
        if entropy_logits.device != ind_logits.device or entropy_logits.shape[-1] != ind_logits.shape[-1]:
            raise ValueError("Entropy and IND logits must share a device and intent space")
        entropy = negative_entropy(entropy_logits)
    return {"loss": ce + alpha * entropy, "ce": ce, "negative_entropy": entropy}


def perturb_tokens(tokens, vocab_size, *, generator, probability=0.5):
    """Replace eligible IDs uniformly in [4,vocab_size); preserve PAD/UNK/BOS/EOS.

    Sampling includes the original token. A replacement can leave its ID unchanged.
    """
    if (tokens.device.type != "cuda" or tokens.dtype != torch.long or tokens.ndim != 2
            or generator.device != tokens.device or vocab_size <= 4
            or ((tokens < 0) | (tokens >= vocab_size)).any()):
        raise ValueError("Perturb requires valid CUDA token IDs, a CUDA RNG, and non-special vocabulary")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    selected = torch.rand(tokens.shape, device=tokens.device, generator=generator) < probability
    replacements = torch.randint(4, vocab_size, tokens.shape, device=tokens.device, generator=generator)
    return torch.where(selected & tokens.ge(4), replacements, tokens)


def select_final_generator(checkpoints: Sequence[Mapping], configured_epochs=80):
    """Select by completed epoch only, without validation or OOD metrics."""
    if type(configured_epochs) is not int or configured_epochs <= 0:
        raise ValueError("configured_epochs must be positive")
    matches = [item for item in checkpoints if item.get("epoch") == configured_epochs]
    if len(matches) != 1:
        raise ValueError("Exactly one final configured generator epoch checkpoint is required")
    return matches[0]


@dataclass(frozen=True)
class GeneratedCorpus:
    """Text plus checked provenance, supplied by the generation artifact reader.

    Required manifest keys: variant, epoch, epochs, checkpoint_sha256,
    data_sha256, vocabulary_sha256, count. No intent targets are accepted.
    """

    records: tuple[UnlabeledRecord, ...]
    manifest: dict


@dataclass(frozen=True)
class EntropySource:
    variant: str
    records: tuple[UnlabeledRecord, ...]
    manifest: dict


def _unlabeled(records):
    if not records or any(type(row) is not UnlabeledRecord or not row.text.strip() for row in records):
        raise ValueError("Entropy source must contain nonempty UnlabeledRecord text only, with no labels")
    if len({row.id for row in records}) != len(records):
        raise ValueError("Entropy record IDs must be unique (duplicate text is allowed)")


def _ind_records(records, partition, num_classes):
    if not records or any(
        type(row) is not Record or row.source != "full" or row.partition != partition
        or row.is_ood or type(row.target) is not int or not 0 <= row.target < num_classes
        or not row.text.strip() for row in records
    ):
        raise ValueError(f"Expected labeled full:{partition} IND records only; test/OOD labels are forbidden")
    if len({row.id for row in records}) != len(records):
        raise ValueError("IND record IDs must be unique")


def make_entropy_source(config: ClassifierConfig, vocabulary: Vocabulary, *,
                        train: Sequence[Record], dmix: Mixture | None = None,
                        generated: GeneratedCorpus | None = None,
                        data_sha256: str | None = None,
                        checkpoint_sha256: str | None = None) -> EntropySource | None:
    """Resolve exactly one source. Reject missing, extra, or incompatible inputs."""
    _ind_records(train, "train", config.num_classes)
    if config.variant == "MSP":
        if dmix is not None or generated is not None:
            raise ValueError("MSP has no entropy source")
        return None
    if config.variant == "Perturb":
        if dmix is not None or generated is not None or len(vocabulary.tokens) <= 4:
            raise ValueError("Perturb uses only IND train text and needs non-special vocabulary")
        return EntropySource("Perturb", tuple(UnlabeledRecord(row.id, row.text) for row in train),
                             {"probability": config.perturb_probability, "seed": derived_seed(config.seed, "perturbation")})
    if config.variant == "Mix":
        if dmix is None or generated is not None:
            raise ValueError("Mix requires Dmix and forbids generated text")
        _unlabeled(dmix.records)
        ids = [row.id for row in dmix.records]
        train_text = {row.id: row.text for row in train}
        ind_count = sum(row.id.startswith("full:train:") for row in dmix.records)
        ood_count = sum(row.id.startswith("plus:oos_train:") for row in dmix.records)
        if (not dmix.manifest.get("labels_hidden") or dmix.manifest.get("ids") != ids
                or dmix.manifest.get("membership_sha256") != content_hash(sorted(ids))
                or dmix.manifest.get("ordered_ids_sha256") != content_hash(ids)
                or any(not (row.id.startswith("full:train:") or row.id.startswith("plus:oos_train:"))
                       for row in dmix.records)
                or dmix.manifest.get("ind_count") != ind_count
                or dmix.manifest.get("ood_count") != ood_count
                or any(row.id.startswith("full:train:") and train_text.get(row.id) != row.text
                       for row in dmix.records)):
            raise ValueError("Dmix membership/provenance must contain training IDs only with hidden labels")
        if config.paper and (dmix.manifest.get("ind_count") != 10000 or dmix.manifest.get("ood_count") != 250):
            raise ValueError("Paper Mix requires 10000 IND and 250 OOD training records")
        return EntropySource("Mix", tuple(dmix.records), dict(dmix.manifest))
    if generated is None or dmix is not None:
        raise ValueError(f"{config.variant} requires an exported text corpus, not Dmix or latent codes")
    _unlabeled(generated.records)
    metadata = generated.manifest
    if (not data_sha256 or not checkpoint_sha256
            or metadata.get("data_sha256") != data_sha256
            or metadata.get("checkpoint_sha256") != checkpoint_sha256
            or metadata.get("vocabulary_sha256") != vocabulary.sha256
            or metadata.get("variant") != config.variant
            or metadata.get("epoch") != config.generator_epochs
            or metadata.get("epochs") != config.generator_epochs
            or metadata.get("count") != len(generated.records)):
        raise ValueError("Generated corpus data/checkpoint/vocabulary/variant/final-epoch provenance mismatch")
    if config.paper and len(generated.records) != 15000:
        raise ValueError("Paper generated entropy corpus requires 15000 text records")
    return EntropySource(config.variant, tuple(generated.records), dict(metadata))


@torch.no_grad()
def predict_confidence(model: SentenceCNN, tokens: torch.Tensor) -> dict:
    """Return CUDA logits, intent IDs and maximum-softmax confidence, not OOD labels."""
    training = model.training
    model.eval()
    try:
        logits = model(tokens)
        _logits(logits)
        confidence, prediction = logits.softmax(-1).max(-1)
        return {"logits": logits, "prediction": prediction, "confidence": confidence}
    finally:
        model.train(training)


@dataclass(frozen=True)
class EpochEvent:
    epoch: int
    selected: bool
    final: bool
    metrics: dict


@dataclass(frozen=True)
class ClassifierResult:
    selected_epoch: int
    final_epoch: int
    best_validation_accuracy: float
    selected_state: dict
    final_state: dict
    history: tuple[dict, ...]


class FinalClassifierTrainer:
    """Train a fresh CNN and expose state to reproduce.state checkpoint callbacks.

    Epochs use max(IND batches, entropy batches) updates. Each shorter loader
    restarts and reshuffles independently. Callback runs after completed epochs.
    """

    def __init__(self, config: ClassifierConfig, vocabulary: Vocabulary,
                 embeddings: EmbeddingResult | None = None):
        self.config, self.vocabulary = config, vocabulary
        self.model = build_final_classifier(config, vocabulary, embeddings)
        self.device = self.model.embedding.weight.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, capturable=True)
        self.generators = {
            "ind_loader": torch.Generator(device=self.device).manual_seed(derived_seed(config.seed, "loader")),
            "entropy_loader": torch.Generator(device=self.device).manual_seed(derived_seed(config.seed + 1, "loader")),
            "perturbation": torch.Generator(device=self.device).manual_seed(derived_seed(config.seed, "perturbation")),
        }
        self.completed_epoch = self.selected_epoch = 0
        self.best_validation_accuracy = -1.0
        self.selected_state = None

    def selection_state(self):
        """Store this value in save_checkpoint(extra=...) at an epoch boundary."""
        return {"completed_epoch": self.completed_epoch, "selected_epoch": self.selected_epoch,
                "best_validation_accuracy": self.best_validation_accuracy,
                "selected_state": self.selected_state}

    def restore_selection(self, state):
        """Call after reproduce.state.restore_checkpoint restores model/optimizer/RNG."""
        completed, selected = state["completed_epoch"], state["selected_epoch"]
        accuracy = state["best_validation_accuracy"]
        if (type(completed) is not int or type(selected) is not int
                or not 1 <= selected <= completed <= self.config.epochs
                or not math.isfinite(accuracy) or not 0 <= accuracy <= 1
                or state["selected_state"] is None):
            raise ValueError("Invalid classifier epoch selection state")
        expected = self.model.state_dict()
        saved = state["selected_state"]
        if (set(saved) != set(expected) or any(
            saved[key].shape != value.shape or saved[key].device != self.device
            or not torch.isfinite(saved[key]).all() for key, value in expected.items()
        )):
            raise ValueError("Selected classifier weights do not match this CUDA model")
        self.completed_epoch, self.selected_epoch = completed, selected
        self.best_validation_accuracy = accuracy
        self.selected_state = {key: value.detach().clone() for key, value in saved.items()}

    def _encode(self, records):
        return torch.tensor([encode_text(row.text, self.vocabulary, max_length=self.config.max_length).ids
                             for row in records], dtype=torch.long, device=self.device)

    def _batches(self, size, stream):
        while True:
            order = torch.randperm(size, device=self.device, generator=self.generators[stream])
            for start in range(0, size, self.config.batch_size):
                yield order[start:start + self.config.batch_size]

    def _snapshot(self):
        return {key: value.detach().clone() for key, value in self.model.state_dict().items()}

    def fit(self, train: Sequence[Record], validation: Sequence[Record], *,
            source: EntropySource | None = None,
            on_epoch: Callable[[EpochEvent, "FinalClassifierTrainer"], None] | None = None) -> ClassifierResult:
        """No test data or evaluation metrics are accepted by the selection API."""
        _ind_records(train, "train", self.config.num_classes)
        _ind_records(validation, "val", self.config.num_classes)
        if self.config.variant == "MSP":
            if source is not None:
                raise ValueError("MSP has no entropy source")
        elif source is None or source.variant != self.config.variant:
            raise ValueError("An entropy source matching the configured variant is required")
        else:
            # Revalidate public source values instead of trusting their constructor.
            if source.variant == "Perturb":
                checked = make_entropy_source(self.config, self.vocabulary, train=train)
            elif source.variant == "Mix":
                checked = make_entropy_source(self.config, self.vocabulary, train=train,
                                              dmix=Mixture(source.records, source.manifest))
            else:
                checked = make_entropy_source(
                    self.config, self.vocabulary, train=train,
                    generated=GeneratedCorpus(source.records, source.manifest),
                    data_sha256=source.manifest.get("data_sha256"),
                    checkpoint_sha256=source.manifest.get("checkpoint_sha256"))
            if source.records != checked.records or source.manifest != checked.manifest:
                raise ValueError("Entropy source does not match its declared policy")
        train_tokens, val_tokens = self._encode(train), self._encode(validation)
        targets = torch.tensor([row.target for row in train], device=self.device, dtype=torch.long)
        val_targets = torch.tensor([row.target for row in validation], device=self.device, dtype=torch.long)
        entropy_tokens = None if source is None else self._encode(source.records)
        ind_batches = math.ceil(len(train) / self.config.batch_size)
        entropy_batches = 0 if source is None else math.ceil(len(source.records) / self.config.batch_size)
        steps = max(ind_batches, entropy_batches)
        history = []
        for epoch in range(self.completed_epoch + 1, self.config.epochs + 1):
            self.model.train()
            ind_loader = self._batches(len(train), "ind_loader")
            entropy_loader = None if source is None else self._batches(len(source.records), "entropy_loader")
            ce_sum = entropy_sum = 0.0
            ind_seen = entropy_seen = 0
            for _ in range(steps):
                ids = next(ind_loader)
                er_logits = None
                if entropy_loader is not None:
                    entropy_ids = next(entropy_loader)
                    batch = entropy_tokens[entropy_ids]
                    if source.variant == "Perturb":
                        batch = perturb_tokens(batch, len(self.vocabulary.tokens),
                                               generator=self.generators["perturbation"],
                                               probability=self.config.perturb_probability)
                    er_logits = self.model(batch)
                    entropy_seen += len(entropy_ids)
                self.optimizer.zero_grad(set_to_none=True)
                losses = classifier_objective(self.model(train_tokens[ids]), targets[ids],
                                              er_logits, alpha=self.config.alpha)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError("Non-finite final-classifier objective")
                losses["loss"].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise RuntimeError("Non-finite final-classifier gradient")
                self.optimizer.step()
                ce_sum += losses["ce"].item() * len(ids)
                if er_logits is not None:
                    entropy_sum += losses["negative_entropy"].item() * len(entropy_ids)
                ind_seen += len(ids)
            correct = 0
            for start in range(0, len(validation), self.config.batch_size):
                predictions = predict_confidence(self.model, val_tokens[start:start + self.config.batch_size])
                correct += predictions["prediction"].eq(val_targets[start:start + self.config.batch_size]).sum().item()
            accuracy = correct / len(validation)
            selected = accuracy > self.best_validation_accuracy  # Strict comparison keeps the earliest tie.
            if selected:
                self.selected_epoch, self.best_validation_accuracy = epoch, accuracy
                self.selected_state = self._snapshot()
            self.completed_epoch = epoch
            metrics = {"epoch": epoch, "ce": ce_sum / ind_seen,
                       "negative_entropy": entropy_sum / entropy_seen if entropy_seen else 0.0,
                       "validation_accuracy": accuracy, "updates": steps,
                       "ind_examples": ind_seen, "entropy_examples": entropy_seen,
                       "ind_restarts": (steps - 1) // ind_batches,
                       "entropy_restarts": (steps - 1) // entropy_batches if entropy_batches else 0}
            history.append(metrics)
            if on_epoch is not None:
                on_epoch(EpochEvent(epoch, selected, epoch == self.config.epochs, metrics), self)
        return ClassifierResult(self.selected_epoch, self.completed_epoch, self.best_validation_accuracy,
                                self.selected_state, self._snapshot(), tuple(history))
