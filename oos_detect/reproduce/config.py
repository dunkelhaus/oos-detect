"""Resolved experiment profiles. This module imports only the standard library."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping, Sequence


VARIANTS = ("MSP", "Perturb", "Mix", "POG", "AEPOG")
MODES = ("pog", "aepog", "pog-no-noise", "pog-no-soft-token")
PROFILES = ("paper", "tiny-smoke-non-paper")
STAGES = ("prepare", "train-generator", "generate", "train-classifier", "evaluate")


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _number(name, value, *, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class ReproductionConfig:
    """One variant's campaign; use for_seed() to resolve one run explicitly."""

    profile: str = "paper"
    variant: str = "POG"
    mode: str = "pog"
    seeds: tuple[int, ...] = tuple(range(10))
    device: str = "cuda:0"
    generator_epochs: int = 80
    classifier_epochs: int = 30
    batch_size: int = 32
    embedding_dim: int = 100
    code_dim: int = 100
    noise_dim: int = 100
    decoder_hidden_dim: int = 100
    gan_hidden_dim: int = 512
    gan_layers: int = 4
    num_classes: int = 150
    out_channels: int = 128
    kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    mlp_hidden_size: int = 512
    max_length: int = 64
    noise_std: float = 1.0
    ae_lr: float = 1e-3
    auxiliary_lr: float = 1e-3
    generator_lr: float = 1e-4
    critic_lr: float = 1e-4
    classifier_lr: float = 1e-3
    gan_betas: tuple[float, float] = (0.0, 0.9)
    gp_weight: float = 10.0
    entropy_weight: float = 1.0
    alpha: float = 1.0
    perturb_probability: float = 0.5
    ae_clip_norm: float = 5.0
    critic_iterations: int = 5
    dmix_reconstruction_iterations: int = 1
    dmix_critic_iterations: int = 1
    temperature_start: float = 1.0
    temperature_floor: float = 0.1
    baseline_decay: float = 0.9
    generated_count: int = 15000
    generation_temperature: float = 1.0
    empty_retry_limit: int = 100
    dmix_ind_count: int = 10000
    dmix_ood_count: int = 250
    vocabulary_profile: str = "paper"
    random_only: bool = False
    train_limit: int | None = None
    validation_limit: int | None = None

    def __post_init__(self):
        if self.profile not in PROFILES or self.variant not in VARIANTS or self.mode not in MODES:
            raise ValueError("Unknown profile, classifier variant, or generator mode")
        if (self.variant == "AEPOG") != (self.mode == "aepog"):
            raise ValueError("AEPOG variant and aepog mode must be selected together")
        if self.mode.startswith("pog-no-") and self.variant != "POG":
            raise ValueError("POG ablations require the POG classifier variant; Dmix is excluded")
        if (not isinstance(self.seeds, tuple) or not self.seeds
                or any(type(s) is not int or not 0 <= s < 2**63 - 1 for s in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be a nonempty tuple of unique integers in [0, 2**63 - 1)")
        if not isinstance(self.device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("CUDA is required; CPU fallback is disabled")
        integer_names = ("generator_epochs", "classifier_epochs", "batch_size", "embedding_dim",
                         "code_dim", "noise_dim", "decoder_hidden_dim", "gan_hidden_dim", "gan_layers",
                         "num_classes", "out_channels", "mlp_hidden_size", "max_length",
                         "critic_iterations", "dmix_reconstruction_iterations", "dmix_critic_iterations",
                         "generated_count", "empty_retry_limit", "dmix_ind_count", "dmix_ood_count")
        for name in integer_names:
            _positive_integer(name, getattr(self, name))
        if self.num_classes < 2 or self.max_length < 3 or self.gan_layers < 2:
            raise ValueError("Need at least two intents, BOS/content/EOS, and two GAN layers")
        if not isinstance(self.kernel_sizes, tuple) or not self.kernel_sizes:
            raise ValueError("kernel_sizes must be a nonempty tuple")
        for kernel in self.kernel_sizes:
            _positive_integer("kernel_sizes", kernel)
        for name in ("ae_lr", "auxiliary_lr", "generator_lr", "critic_lr", "classifier_lr",
                     "gp_weight", "ae_clip_norm", "temperature_start", "temperature_floor",
                     "generation_temperature"):
            _number(name, getattr(self, name))
        for name in ("noise_std", "entropy_weight", "alpha"):
            _number(name, getattr(self, name), zero=True)
        if self.temperature_floor > self.temperature_start:
            raise ValueError("temperature_floor cannot exceed temperature_start")
        for name in ("baseline_decay", "perturb_probability"):
            value = getattr(self, name)
            _number(name, value, zero=True)
            if value > 1 or (name == "baseline_decay" and value == 1):
                raise ValueError(f"{name} is outside its probability range")
        if not isinstance(self.gan_betas, tuple) or len(self.gan_betas) != 2:
            raise ValueError("gan_betas must contain two Adam coefficients")
        for beta in self.gan_betas:
            _number("gan_betas", beta, zero=True)
            if beta >= 1:
                raise ValueError("gan_betas must be in [0, 1)")
        if type(self.random_only) is not bool or self.vocabulary_profile not in {"paper", "train-only"}:
            raise ValueError("Invalid embedding or vocabulary policy")
        for name in ("train_limit", "validation_limit"):
            if getattr(self, name) is not None:
                _positive_integer(name, getattr(self, name))
        if self.paper:
            # Compare against declared defaults without constructing another config.
            exceptions = {"profile", "variant", "mode", "device", "empty_retry_limit"}
            for field in fields(self):
                if field.name not in exceptions and getattr(self, field.name) != field.default:
                    raise ValueError(f"Paper profile requires {field.name}={field.default!r}; use a non-paper profile for overrides")
        elif not self.random_only:
            raise ValueError("tiny-smoke-non-paper explicitly requires random_only=True")

    @property
    def paper(self):
        return self.profile == "paper"

    @property
    def uses_generator(self):
        return self.variant in {"POG", "AEPOG"}

    @property
    def uses_dmix(self):
        return self.variant in {"Mix", "AEPOG"}

    def to_dict(self):
        """Return JSON-compatible resolved settings, with no hidden overrides."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def model_kwargs(self):
        names = ("embedding_dim", "code_dim", "noise_dim", "decoder_hidden_dim", "gan_hidden_dim",
                 "gan_layers", "num_classes", "out_channels", "kernel_sizes", "mlp_hidden_size",
                 "noise_std", "device")
        return {name: getattr(self, name) for name in names}

    def pog_settings(self):
        """Keyword arguments for pseudo_ood_generation.trainer.POGSettings."""
        names = ("mode", "batch_size", "ae_lr", "auxiliary_lr", "generator_lr", "critic_lr",
                 "gan_betas", "gp_weight", "entropy_weight", "ae_clip_norm", "critic_iterations",
                 "dmix_reconstruction_iterations", "dmix_critic_iterations", "temperature_start",
                 "temperature_floor", "baseline_decay")
        return {**{name: getattr(self, name) for name in names},
                "epochs": self.generator_epochs, "max_steps": self.max_length - 1}

    def classifier_settings(self, seed):
        """Keyword arguments for reproduce.classifier.ClassifierConfig."""
        if type(seed) is not int or seed not in self.seeds:
            raise ValueError("Select an explicitly declared campaign seed")
        names = ("variant", "generator_epochs", "batch_size", "alpha", "perturb_probability",
                 "embedding_dim", "num_classes", "out_channels", "kernel_sizes",
                 "mlp_hidden_size", "max_length", "device")
        return {**{name: getattr(self, name) for name in names}, "paper": self.paper,
                "epochs": self.classifier_epochs, "learning_rate": self.classifier_lr, "seed": seed}

    def for_seed(self, seed):
        """Resolve manifest settings for state.create_run; preserve the campaign seeds."""
        self.classifier_settings(seed)
        return {**self.to_dict(), "seed": seed}


def resolve_config(values=None):
    """Resolve defaults, reject unknown keys, and validate without loading models."""
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {field.name for field in fields(ReproductionConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    values = dict(values)
    for name in ("seeds", "kernel_sizes", "gan_betas"):
        if name in values:
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"{name} must be an array")
            values[name] = tuple(values[name])
    return ReproductionConfig(**values)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def load_config(path):
    with Path(path).open(encoding="utf-8") as source:
        values = json.load(source, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return resolve_config(values)


def paper_config(variant="POG", *, mode=None, device="cuda:0"):
    return ReproductionConfig(variant=variant, mode=mode or ("aepog" if variant == "AEPOG" else "pog"), device=device)


def smoke_config(variant="POG", *, mode=None, device="cuda:0"):
    """Tiny, explicitly non-paper settings; embeddings stay 100d for data.load_embeddings."""
    return replace(paper_config(variant, mode=mode, device=device),
                   profile="tiny-smoke-non-paper", seeds=(0,), generator_epochs=1,
                   classifier_epochs=1, batch_size=2, code_dim=8, noise_dim=8,
                   decoder_hidden_dim=8, gan_hidden_dim=16, num_classes=2,
                   out_channels=4, mlp_hidden_size=8, max_length=8, critic_iterations=1,
                   generated_count=8, dmix_ind_count=4, dmix_ood_count=2,
                   random_only=True, train_limit=8, validation_limit=4)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_prerequisites(config, stage, *, embedding_path=None, generated_manifest=None,
                           generated_records=None, data_sha256=None, vocabulary_sha256=None,
                           checkpoint_sha256=None):
    """Check prerequisites before model creation. No downloads or training occur.

    Generated records are mappings from the artifact reader, not latent tensors.
    The reader must verify the artifact file hashes before calling this function.
    """
    if not isinstance(config, ReproductionConfig) or stage not in STAGES:
        raise ValueError("Expected a resolved config and a supported stage")
    if stage in {"train-generator", "generate"} and not config.uses_generator:
        raise ValueError(f"{config.variant} has no generator stage")
    if stage in {"train-generator", "train-classifier"}:
        if config.paper and (embedding_path is None or not Path(embedding_path).is_file()):
            raise FileNotFoundError("Paper training requires a supplied glove.6B.100d.txt file; no download or random-vector fallback is allowed")
        if config.random_only and embedding_path is not None:
            raise ValueError("Random-only smoke forbids a pretrained embedding path")
    if stage != "train-classifier":
        return
    if not config.uses_generator:
        if generated_manifest is not None or generated_records is not None:
            raise ValueError(f"{config.variant} forbids a generated entropy corpus")
        return
    if not isinstance(generated_manifest, Mapping) or generated_records is None:
        raise ValueError("Generate and load a separate pseudo-OOD text corpus before classifier training")
    manifest = generated_manifest
    # Artifact profiles identify their stage, not the campaign configuration.
    expected_profile = "paper-generation" if config.paper else "tiny-non-paper-generation"
    for name, expected in (("data_sha256", data_sha256), ("vocabulary_sha256", vocabulary_sha256),
                           ("checkpoint_sha256", checkpoint_sha256)):
        if not _sha256(expected) or manifest.get(name) != expected:
            raise ValueError(f"Generated corpus {name} mismatch or missing expected SHA-256")
    if (any(type(manifest.get(name)) is not int for name in ("epoch", "epochs", "count"))
            or manifest.get("variant") != config.variant or manifest.get("mode") != config.mode
            or manifest.get("epoch") != config.generator_epochs
            or manifest.get("epochs") != config.generator_epochs
            or manifest.get("count") != config.generated_count
            or manifest.get("profile") != expected_profile):
        raise ValueError("Generated corpus variant/mode/profile/final-epoch/count mismatch")
    if (not isinstance(generated_records, Sequence) or isinstance(generated_records, (str, bytes))
            or len(generated_records) != config.generated_count):
        raise ValueError("Generated corpus must contain the configured number of text records")
    ids = set()
    for row in generated_records:
        if (not isinstance(row, Mapping) or set(row) != {"id", "text"}
                or not isinstance(row["id"], str) or not row["id"]
                or row["id"] in ids or not isinstance(row["text"], str) or not row["text"].strip()):
            raise ValueError("Generated records require unique IDs and nonempty text only; no labels or latent codes")
        ids.add(row["id"])
