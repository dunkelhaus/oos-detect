"""CPU statistical reports from precomputed predictions; no model execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import canonical_json, content_hash


METRICS = ("IND-accuracy", "AUROC", "AUPR-IND", "AUPR-OOD",
           "FPR95-IND", "FPR90-IND", "FPR95-OOD", "FPR90-OOD")
CONVENTIONS = {
    "IND": "positive=IND; score=c=max softmax; predict positive if score >= threshold",
    "OOD": "positive=OOD; score=1-c; predict positive if score >= threshold",
    "AUROC": "OOD-positive with 1-c (equivalent to IND-positive with c)",
    "AUPR": "non-interpolated average precision; sum(delta recall * precision)",
    "ROC": "all tied scores grouped; first observed threshold reaching target TPR; no interpolation",
    "units": "fraction in [0,1]",
}
ASSUMPTIONS = (
    "Paper classifier vocabulary uses declared train/validation/test text, without held-out labels (transductive).",
    "Generator vocabulary uses IND training text only.",
    "Engineering defaults and pretrained embedding choice are assumptions, not paper claims.",
    "Paper AUPR is designated OOD-positive; FPR orientation is ambiguous; comparisons are provisional.",
    "AUPR depends on class balance; CLINC test has 4500 IND and 1000 OOD examples.",
)
PAPER_REFERENCE = {
    "paper": "1909.03862v4", "dataset": "OSQ/CLINC", "units": "percent",
    "status": "provisional orientation; no numerical-parity pass/fail gate",
    "AUPR_designation": "OOD-positive; class-balance caveat applies",
    "FPR_orientation": "ambiguous in paper prose; retain both reproduction orientations",
    "values": {
        "MSP": {"AUROC": 92.86, "AUPR": 98.24, "FPR95": 39.72, "FPR90": 21.76, "IND-accuracy": 92.61},
        "POG": {"AUROC": 95.41, "AUPR": 98.94, "FPR95": 25.00, "FPR90": 10.10, "IND-accuracy": 93.31},
        "AEPOG": {"AUROC": 95.83, "AUPR": 99.05, "FPR95": 23.70, "FPR90": 9.50, "IND-accuracy": 93.32},
    },
}


@dataclass(frozen=True)
class EvaluationProvenance:
    """Identity fixed before test evaluation. Hashes identify actual artifacts."""

    seed: int
    variant: str
    data_sha256: str
    vocabulary_sha256: str
    selected_checkpoint: str
    selected_checkpoint_sha256: str
    selection_split: str = "val"
    selection_criterion: str = "best IND-validation accuracy; earliest epoch on ties"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.selection_split != "val":
            raise ValueError("Checkpoint selection must use val, never test")
        for key, value in asdict(self).items():
            if key != "seed" and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Provenance {key} must be nonempty")
            if key.endswith("_sha256") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"Provenance {key} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class PredictionRecord:
    """One source ID with IND-only intent probabilities and a shared identity."""

    id: str
    split: str
    is_ood: bool
    target: int | None
    prediction: int
    probabilities: tuple[float, ...]
    provenance: EvaluationProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or self.split not in {"val", "test"}:
            raise ValueError("Prediction requires a source ID and val/test split")
        if type(self.is_ood) is not bool:
            raise ValueError("is_ood must be a bool")
        if not isinstance(self.provenance, EvaluationProvenance):
            raise ValueError("Prediction requires EvaluationProvenance")
        probs = self.probabilities
        if len(probs) < 2 or any(isinstance(p, bool) or not isinstance(p, (int, float))
                                 or not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
            raise ValueError("Intent probabilities must be finite values in [0,1], with at least two intents")
        if not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Intent probabilities must sum to one")
        if type(self.prediction) is not int or not 0 <= self.prediction < len(probs):
            raise ValueError("Invalid intent prediction")
        if self.prediction != max(range(len(probs)), key=probs.__getitem__):
            raise ValueError("Prediction must be argmax, with lowest intent ID on ties")
        if self.is_ood:
            if self.target is not None:
                raise ValueError("OOD records must not have an intent target")
        elif type(self.target) is not int or not 0 <= self.target < len(probs):
            raise ValueError("IND records require a valid intent target")
        object.__setattr__(self, "probabilities", tuple(float(p) for p in probs))

    @property
    def confidence(self) -> float:
        return max(self.probabilities)

    def to_dict(self) -> dict:
        return {**asdict(self), "confidence": self.confidence, "ood_score": 1 - self.confidence}


def prediction_records(rows: Iterable[Mapping], probabilities: Sequence[Sequence[float]], *,
                       provenance: EvaluationProvenance, split: str) -> tuple[PredictionRecord, ...]:
    """Join prepared source rows to already computed probability rows, in order.

    Source partition must agree with the requested split. No model is loaded.
    """
    rows, probabilities = list(rows), list(probabilities)
    if split not in {"val", "test"}:
        raise ValueError("Prediction split must be val or test")
    if len(rows) != len(probabilities):
        raise ValueError("Source and probability row counts differ")
    result = []
    for row, probs in zip(rows, probabilities):
        if row.get("partition", row.get("split")) not in {split, f"oos_{split}"}:
            raise ValueError("Source split provenance mismatch")
        if "partition" in row and row["partition"].startswith("oos_") != row["is_ood"]:
            raise ValueError("Source partition/status provenance mismatch")
        probs = tuple(probs)
        if not probs:
            raise ValueError("Empty probability vector")
        prediction = max(range(len(probs)), key=probs.__getitem__)
        result.append(PredictionRecord(row["id"], split, row["is_ood"], row["target"],
                                       prediction, probs, provenance))
    return tuple(result)


def _validate(records: Iterable[PredictionRecord]) -> tuple[PredictionRecord, ...]:
    rows = tuple(records)
    if any(not isinstance(r, PredictionRecord) for r in rows):
        raise ValueError("Evaluation requires PredictionRecord objects")
    if not rows or {r.is_ood for r in rows} != {False, True}:
        raise ValueError("Evaluation requires both IND and OOD examples")
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate source IDs in evaluation")
    first = rows[0]
    if any(r.provenance != first.provenance or r.split != first.split
           or len(r.probabilities) != len(first.probabilities) for r in rows):
        raise ValueError("Prediction provenance, split, or intent dimension mismatch")
    return rows


def _operating_point(rows: Sequence[PredictionRecord], positive: str, target: float) -> dict:
    if (positive not in {"IND", "OOD"} or isinstance(target, bool)
            or not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= 1):
        raise ValueError("Require positive=IND/OOD and target TPR in (0,1]")
    labels = [int(r.is_ood == (positive == "OOD")) for r in rows]
    scores = [1 - r.confidence if positive == "OOD" else r.confidence for r in rows]
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    index = next(i for i, rate in enumerate(tpr) if rate >= target)
    return {"positive_class": positive, "score": "1-c" if positive == "OOD" else "c",
            "comparison": ">=", "target_tpr": target, "tpr": float(tpr[index]),
            "fpr": float(fpr[index]), "threshold": float(thresholds[index])}


def calibrate_threshold(records: Iterable[PredictionRecord], *, positive: str = "OOD",
                        target_tpr: float = 0.95) -> dict:
    """Select a deployable threshold using validation records only."""
    rows = _validate(records)
    if rows[0].split != "val":
        raise ValueError("Deployment threshold calibration requires val, never test")
    return {"kind": "validation-calibrated-deployment", "split": "val",
            "provenance": asdict(rows[0].provenance),
            "validation_records_sha256": content_hash([r.to_dict() for r in rows]),
            "conventions": dict(CONVENTIONS), **_operating_point(rows, positive, target_tpr)}


def _check_calibration(calibration: Mapping, provenance: EvaluationProvenance) -> None:
    if (calibration.get("kind") != "validation-calibrated-deployment"
            or calibration.get("split") != "val"
            or calibration.get("provenance") != asdict(provenance)
            or not calibration.get("validation_records_sha256")
            or calibration.get("conventions") != CONVENTIONS):
        raise ValueError("Threshold provenance mismatch or non-validation calibration")
    digest = calibration["validation_records_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid validation calibration digest")
    if (calibration.get("positive_class") not in {"IND", "OOD"}
            or calibration.get("comparison") != ">="
            or calibration.get("score") != ("1-c" if calibration["positive_class"] == "OOD" else "c")
            or isinstance(calibration.get("threshold"), bool)
            or not isinstance(calibration.get("threshold"), (int, float))
            or not math.isfinite(calibration["threshold"]) or not 0 <= calibration["threshold"] <= 1):
        raise ValueError("Invalid calibrated threshold")
    for key in ("target_tpr", "tpr", "fpr"):
        value = calibration.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Invalid calibrated operating point")
    if not 0 < calibration["target_tpr"] <= calibration["tpr"]:
        raise ValueError("Invalid calibrated target TPR")


def evaluate_records(records: Iterable[PredictionRecord], *, deployment_threshold: Mapping | None = None,
                     assumptions: Sequence[str] = (), deviations: Sequence[str] = ()) -> dict:
    """Return metrics in fractions and separate descriptive ROC operating points."""
    rows = _validate(records)
    ind = [r for r in rows if not r.is_ood]
    labels = [int(r.is_ood) for r in rows]
    confidence = [r.confidence for r in rows]
    metrics = {
        "IND-accuracy": sum(r.prediction == r.target for r in ind) / len(ind),
        "AUROC": float(roc_auc_score(labels, [1 - c for c in confidence])),
        "AUPR-IND": float(average_precision_score([1 - y for y in labels], confidence, pos_label=1)),
        "AUPR-OOD": float(average_precision_score(labels, [1 - c for c in confidence], pos_label=1)),
    }
    points = {}
    for positive in ("IND", "OOD"):
        for percent in (95, 90):
            key = f"FPR{percent}-{positive}"
            points[key] = _operating_point(rows, positive, percent / 100)
            metrics[key] = points[key]["fpr"]
    report = {"schema_version": 1, "provenance": asdict(rows[0].provenance),
              "split": rows[0].split, "metrics": metrics, "conventions": dict(CONVENTIONS),
              "class_balance": {"IND": len(ind), "OOD": len(rows) - len(ind)},
              "roc_operating_points": {"kind": "descriptive-split-ROC-not-deployment",
                                       "split": rows[0].split, "points": points},
              "assumptions": list(ASSUMPTIONS) + list(assumptions), "deviations": list(deviations),
              "deployment_threshold": None}
    if deployment_threshold is not None:
        _check_calibration(deployment_threshold, rows[0].provenance)
        positive = deployment_threshold["positive_class"]
        labels_positive = [r.is_ood == (positive == "OOD") for r in rows]
        scores = [1 - r.confidence if positive == "OOD" else r.confidence for r in rows]
        decisions = [score >= deployment_threshold["threshold"] for score in scores]
        report["deployment_threshold"] = {
            "calibration": dict(deployment_threshold),
            "application_split": rows[0].split,
            "positive_class": positive, "score": deployment_threshold["score"],
            "comparison": ">=", "threshold": deployment_threshold["threshold"],
            "tpr": sum(d and y for d, y in zip(decisions, labels_positive)) / sum(labels_positive),
            "fpr": sum(d and not y for d, y in zip(decisions, labels_positive)) / sum(not y for y in labels_positive)}
    return report


def save_evaluation(output_dir: str | Path, records: Iterable[PredictionRecord], **kwargs) -> dict:
    """Write predictions and summary in a fresh directory; never overwrite a run."""
    rows = tuple(records)
    report = evaluate_records(rows, **kwargs)
    payload = "".join(json.dumps(r.to_dict(), sort_keys=True, allow_nan=False) + "\n" for r in rows).encode()
    report["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("xb") as stream:
        stream.write(payload)
    with (output / "summary.json").open("xb") as stream:
        stream.write(canonical_json(report))
    return report


def aggregate_reports(reports: Iterable[Mapping], *, expected_seeds: Iterable[int] = range(10)) -> dict:
    """Aggregate one comparable variant; retain each seed and use sample SD."""
    reports, expected = list(reports), list(expected_seeds)
    if (not expected or any(type(s) is not int or s < 0 for s in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("Expected seeds must be distinct nonnegative integers")
    seen, per_seed = set(), {}
    identity = None
    for report in reports:
        provenance = EvaluationProvenance(**report["provenance"])
        seed = provenance.seed
        if seed in seen or seed not in expected:
            raise ValueError("Duplicate or unexpected seed")
        seen.add(seed)
        comparable = {key: report[key] for key in ("split", "conventions", "class_balance", "assumptions", "deviations")}
        comparable["provenance"] = {key: value for key, value in asdict(provenance).items()
                                    if key not in {"seed", "selected_checkpoint", "selected_checkpoint_sha256"}}
        if report["conventions"] != CONVENTIONS or report["split"] != "test":
            raise ValueError("Aggregation requires test reports with matching metric conventions")
        if identity is not None and comparable != identity:
            raise ValueError("Aggregation provenance or assumptions mismatch")
        identity = comparable
        values = report["metrics"]
        if set(values) != set(METRICS) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                             or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError("Require all explicitly oriented finite metrics in [0,1]")
        per_seed[str(seed)] = dict(report)
    statistics = {}
    for metric in METRICS:
        values = [per_seed[str(s)]["metrics"][metric] for s in sorted(seen)]
        statistics[metric] = {"n": len(values), "mean": mean(values) if values else None,
                              "sample_sd": stdev(values) if len(values) > 1 else None}
    variant = identity["provenance"]["variant"] if identity else None
    reference_key = {"ER+POG": "POG", "ER+AEPOG": "AEPOG"}.get(variant, variant)
    reference = PAPER_REFERENCE["values"].get(reference_key)
    differences = {}
    if reference:
        for metric in METRICS:
            paper_metric = "AUPR" if metric.startswith("AUPR-") else metric.split("-")[0] if metric.startswith("FPR") else metric
            value = statistics[metric]["mean"]
            orientation = metric.rsplit("-", 1)[-1] if metric.startswith(("FPR", "AUPR-")) else None
            status = "provisional; no numerical-parity pass/fail gate"
            if metric == "AUPR-IND":
                status = "orientation mismatch with paper OOD designation; not a like-for-like comparison"
            elif metric.startswith("FPR"):
                status = "paper positive class ambiguous; both reproduction orientations retained"
            differences[metric] = {
                "paper_metric": paper_metric, "paper_percent": reference[paper_metric],
                "reproduction_positive_class": orientation,
                "paper_positive_class": "OOD" if metric.startswith("AUPR-") else "ambiguous" if metric.startswith("FPR") else None,
                "mean_minus_paper_percentage_points": 100 * value - reference[paper_metric],
                "status": status}
    return {"schema_version": 1, "variant": variant,
            "status": "complete" if seen == set(expected) else "incomplete",
            "expected_seeds": sorted(expected), "completed_seeds": sorted(seen),
            "missing_seeds": sorted(set(expected) - seen), "per_seed": per_seed,
            "statistics": statistics, "standard_deviation": "sample; ddof=1; null for fewer than two seeds",
            "assumptions": identity["assumptions"] if identity else list(ASSUMPTIONS),
            "deviations": identity["deviations"] if identity else [],
            "comparison_identity": identity, "paper_reference": json.loads(json.dumps(PAPER_REFERENCE)),
            "paper_differences": differences, "numerical_parity_gate": False}
