"""Analytic report checks. These tests do not load or execute a model."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math

import pytest

from oos_detect.reproduce.evaluation import (
    CONVENTIONS, METRICS, EvaluationProvenance, PredictionRecord,
    aggregate_reports, calibrate_threshold, evaluate_records,
    prediction_records, save_evaluation,
)


def provenance(seed=0, variant="MSP"):
    return EvaluationProvenance(
        seed=seed, variant=variant, data_sha256="a" * 64,
        vocabulary_sha256="b" * 64, selected_checkpoint=f"seed-{seed}/selected.pt",
        selected_checkpoint_sha256=hashlib.sha256(str(seed).encode()).hexdigest(),
    )


def records(ind=(0.9, 0.8), ood=(0.6, 0.55), *, split="test", seed=0, variant="MSP"):
    identity = provenance(seed, variant)
    return tuple(
        PredictionRecord(f"source:{split}:{status}:{i}", split, status,
                         None if status else 0, 0, (c, 1 - c), identity)
        for status, scores in ((False, ind), (True, ood))
        for i, c in enumerate(scores)
    )


def test_perfect_separation_and_explicit_names():
    report = evaluate_records(records())
    assert set(report["metrics"]) == set(METRICS)
    for metric in ("IND-accuracy", "AUROC", "AUPR-IND", "AUPR-OOD"):
        assert report["metrics"][metric] == pytest.approx(1)
    for positive in ("IND", "OOD"):
        for percent in (90, 95):
            metric = f"FPR{percent}-{positive}"
            assert report["metrics"][metric] == 0
            point = report["roc_operating_points"]["points"][metric]
            assert point["positive_class"] == positive
            assert point["score"] == ("c" if positive == "IND" else "1-c")
            assert point["comparison"] == ">="
            assert point["target_tpr"] == percent / 100
    assert not {"FPR", "FPR90", "FPR95", "AUPR"} & report["metrics"].keys()
    assert report["conventions"] == CONVENTIONS


def test_reversed_separation():
    metrics = evaluate_records(records(ind=(0.6, 0.55), ood=(0.9, 0.8)))["metrics"]
    assert metrics["AUROC"] == 0
    # Each positive is below both negatives: AP = (1/3 + 2/4)/2.
    assert metrics["AUPR-IND"] == pytest.approx(5 / 12)
    assert metrics["AUPR-OOD"] == pytest.approx(5 / 12)
    assert all(metrics[key] == 1 for key in METRICS if key.startswith("FPR"))


def test_all_tied_imbalanced_scores():
    rows = records(ind=(0.75,) * 3, ood=(0.75,))
    report = evaluate_records(rows)
    assert report["class_balance"] == {"IND": 3, "OOD": 1}
    assert report["metrics"]["AUROC"] == 0.5
    assert report["metrics"]["AUPR-IND"] == 0.75
    assert report["metrics"]["AUPR-OOD"] == 0.25
    for key, point in report["roc_operating_points"]["points"].items():
        assert report["metrics"][key] == point["fpr"] == point["tpr"] == 1
        assert point["threshold"] == (0.75 if key.endswith("IND") else 0.25)
    assert evaluate_records(reversed(rows))["metrics"] == report["metrics"]


def test_imbalanced_rankings_have_distinct_orientations():
    metrics = evaluate_records(records(ind=(0.9, 0.8, 0.7), ood=(0.85,)))["metrics"]
    # One of the three IND/OOD pairs has the correct confidence ordering.
    assert metrics["AUROC"] == pytest.approx(1 / 3)
    assert metrics["AUPR-IND"] == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert metrics["AUPR-OOD"] == pytest.approx(1 / 3)
    assert metrics["FPR95-IND"] == metrics["FPR90-IND"] == 1
    assert metrics["FPR95-OOD"] == metrics["FPR90-OOD"] == pytest.approx(2 / 3)


def test_first_observed_target_keeps_plateau_start():
    report = evaluate_records(records(ind=(0.9,) * 19 + (0.6,), ood=(0.8, 0.7)))
    for percent in (90, 95):
        point = report["roc_operating_points"]["points"][f"FPR{percent}-IND"]
        assert point["threshold"] == 0.9
        assert point["tpr"] == 0.95
        assert point["fpr"] == 0


def test_ties_crossing_target_are_not_split_or_interpolated():
    rows = records(ind=(0.9,) * 18 + (0.8,) * 2, ood=(0.8,))
    report = evaluate_records(rows)
    points = report["roc_operating_points"]["points"]
    assert points["FPR90-IND"]["threshold"] == 0.9
    assert points["FPR90-IND"]["fpr"] == 0
    assert points["FPR95-IND"]["threshold"] == 0.8
    assert points["FPR95-IND"]["tpr"] == 1
    assert points["FPR95-IND"]["fpr"] == 1


@pytest.mark.parametrize("ind,ood", [((), (0.6,)), ((0.9,), ()), ((), ())])
def test_missing_status_is_rejected(ind, ood):
    with pytest.raises(ValueError, match="both IND and OOD"):
        evaluate_records(records(ind, ood))


def test_ind_accuracy_excludes_ood_and_uses_intent_targets():
    rows = list(records())
    rows[0] = replace(rows[0], target=1)
    assert evaluate_records(rows)["metrics"]["IND-accuracy"] == 0.5


def test_record_join_retains_source_confidence_and_identity():
    rows = [dict(id="full:test:0", partition="test", is_ood=False, target=1),
            dict(id="full:oos_test:0", partition="oos_test", is_ood=True, target=None)]
    result = prediction_records(rows, [[0.2, 0.8], [0.5, 0.5]],
                                provenance=provenance(), split="test")
    assert result[0].prediction == 1
    assert result[0].confidence == 0.8
    assert result[1].prediction == 0  # Lowest intent ID on an argmax tie.
    assert result[1].to_dict()["ood_score"] == 0.5
    assert result[0].to_dict()["provenance"] == asdict(provenance())
    with pytest.raises(ValueError, match="row counts"):
        prediction_records(rows, [[0.2, 0.8]], provenance=provenance(), split="test")
    with pytest.raises(ValueError, match="split provenance"):
        prediction_records(rows, [[0.2, 0.8], [0.5, 0.5]], provenance=provenance(), split="val")
    rows[1]["is_ood"] = False
    with pytest.raises(ValueError, match="partition/status"):
        prediction_records(rows, [[0.2, 0.8], [0.5, 0.5]], provenance=provenance(), split="test")


@pytest.mark.parametrize("probabilities", [(0.8, 0.3), (float("nan"), 0.5),
                                            (float("inf"), 0), (-0.1, 1.1), (True, 0), (1.0,)])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        replace(records()[0], probabilities=probabilities)


def test_record_and_evaluation_identity_checks():
    rows = records()
    with pytest.raises(ValueError, match="argmax"):
        replace(rows[0], prediction=1)
    with pytest.raises(ValueError, match="OOD records"):
        replace(rows[-1], target=0)
    with pytest.raises(ValueError, match="IND records"):
        replace(rows[0], target=None)
    with pytest.raises(ValueError, match="Duplicate source"):
        evaluate_records((*rows, rows[0]))
    for replacement in (replace(rows[-1], split="val"),
                        replace(rows[-1], provenance=provenance(1)),
                        replace(rows[-1], probabilities=(0.6, 0.2, 0.2))):
        with pytest.raises(ValueError, match="mismatch"):
            evaluate_records((*rows[:-1], replacement))
    with pytest.raises(ValueError, match="never test"):
        replace(provenance(), selection_split="test")
    with pytest.raises(ValueError, match="SHA-256"):
        replace(provenance(), selected_checkpoint_sha256="not-a-digest")


@pytest.mark.parametrize("positive", ["IND", "OOD"])
def test_validation_calibration_is_separate_from_test_roc(positive):
    validation = records(ind=(0.9, 0.8), ood=(0.6, 0.65), split="val")
    threshold = calibrate_threshold(validation, positive=positive)
    assert threshold["kind"] == "validation-calibrated-deployment"
    assert threshold["split"] == "val"
    test_rows = records(ind=(0.95, 0.9), ood=(0.7, 0.75))
    report = evaluate_records(test_rows, deployment_threshold=threshold)
    application = report["deployment_threshold"]
    assert application["calibration"] == threshold
    assert application["threshold"] == threshold["threshold"]
    assert application["application_split"] == "test"
    assert application["positive_class"] == positive
    descriptive = report["roc_operating_points"]
    assert descriptive["kind"] == "descriptive-split-ROC-not-deployment"
    assert descriptive["points"][f"FPR95-{positive}"]["threshold"] != threshold["threshold"]
    if positive == "OOD":
        assert application["tpr"] == 0  # Test scores cannot recalibrate the threshold.
        assert descriptive["points"]["FPR95-OOD"]["tpr"] == 1
    with pytest.raises(ValueError, match="val, never test"):
        calibrate_threshold(test_rows)


@pytest.mark.parametrize("change", [
    {"split": "test"}, {"kind": "descriptive-split-ROC-not-deployment"},
    {"validation_records_sha256": ""}, {"validation_records_sha256": "invalid"},
    {"score": "c"}, {"comparison": ">"},
    {"threshold": float("nan")}, {"threshold": True}, {"target_tpr": 0},
    {"tpr": 0.5}, {"fpr": float("inf")},
])
def test_invalid_or_test_threshold_is_rejected(change):
    threshold = calibrate_threshold(records(split="val"))
    threshold.update(change)
    with pytest.raises(ValueError, match="calibrat|Threshold"):
        evaluate_records(records(), deployment_threshold=threshold)


@pytest.mark.parametrize("target", [0, 1.01, True, float("nan")])
def test_invalid_calibration_target_is_rejected(target):
    with pytest.raises(ValueError, match="target TPR"):
        calibrate_threshold(records(split="val"), target_tpr=target)


@pytest.mark.parametrize("field,value", [
    ("seed", 1), ("variant", "ER+POG"), ("data_sha256", "c" * 64),
    ("vocabulary_sha256", "c" * 64), ("selected_checkpoint", "other.pt"),
    ("selected_checkpoint_sha256", "c" * 64),
])
def test_threshold_must_match_selected_checkpoint_and_run(field, value):
    threshold = calibrate_threshold(records(split="val"))
    threshold["provenance"][field] = value
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluate_records(records(), deployment_threshold=threshold)


def test_saved_artifact_is_complete_hashed_and_not_overwritten(tmp_path):
    output = tmp_path / "evaluation"
    rows = records()
    report = save_evaluation(output, rows, assumptions=["tiny analytic fixture"],
                             deviations=["two intents, not paper profile"])
    payload = (output / "predictions.jsonl").read_bytes()
    saved = json.loads((output / "summary.json").read_text())
    assert saved == report
    assert report["predictions_sha256"] == hashlib.sha256(payload).hexdigest()
    predictions = [json.loads(line) for line in payload.splitlines()]
    assert predictions == json.loads(json.dumps([row.to_dict() for row in rows]))
    assert predictions[0]["confidence"] == 0.9
    assert predictions[-1]["is_ood"] is True
    assert "tiny analytic fixture" in report["assumptions"]
    assert report["deviations"] == ["two intents, not paper profile"]
    aggregate = aggregate_reports([saved])
    assert aggregate["status"] == "incomplete"
    assert aggregate["missing_seeds"] == list(range(1, 10))
    assert "tiny analytic fixture" in aggregate["assumptions"]
    assert any("transductive" in item for item in aggregate["assumptions"])
    assert aggregate["deviations"] == report["deviations"]
    with pytest.raises(FileExistsError):
        save_evaluation(output, rows)
    assert (output / "predictions.jsonl").read_bytes() == payload


def test_aggregation_retains_seeds_mean_sample_sd_and_provisional_comparison():
    first = evaluate_records(records(seed=0))
    second = evaluate_records(records(ind=(0.6, 0.55), ood=(0.9, 0.8), seed=1))
    report = aggregate_reports([second, first], expected_seeds=[0, 1, 2])
    assert report["status"] == "incomplete"
    assert report["completed_seeds"] == [0, 1]
    assert report["missing_seeds"] == [2]
    assert report["per_seed"] == {"0": first, "1": second}
    assert report["statistics"]["AUROC"] == pytest.approx(
        {"n": 2, "mean": 0.5, "sample_sd": math.sqrt(0.5)})
    assert report["statistics"]["IND-accuracy"]["sample_sd"] == 0
    comparison = report["paper_differences"]
    assert comparison["AUROC"]["mean_minus_paper_percentage_points"] == pytest.approx(50 - 92.86)
    assert comparison["AUPR-IND"]["paper_positive_class"] == "OOD"
    assert "orientation mismatch" in comparison["AUPR-IND"]["status"]
    assert comparison["FPR95-OOD"]["paper_positive_class"] == "ambiguous"
    assert "ambiguous" in comparison["FPR95-IND"]["status"]
    assert report["numerical_parity_gate"] is False
    assert "pass" not in report and "fail" not in report
    assert aggregate_reports([first, second], expected_seeds=[0, 1])["status"] == "complete"


def test_empty_single_seed_and_default_ten_seed_aggregation():
    empty = aggregate_reports([])
    assert empty["status"] == "incomplete"
    assert empty["missing_seeds"] == list(range(10))
    assert empty["statistics"]["AUROC"] == {"n": 0, "mean": None, "sample_sd": None}
    single = aggregate_reports([evaluate_records(records())])
    assert single["missing_seeds"] == list(range(1, 10))
    assert single["statistics"]["AUROC"] == {"n": 1, "mean": 1, "sample_sd": None}
    complete = aggregate_reports([evaluate_records(records(seed=s)) for s in range(10)])
    assert complete["status"] == "complete"
    assert complete["missing_seeds"] == []


@pytest.mark.parametrize("variant,reference", [("MSP", "MSP"), ("ER+POG", "POG"),
                                               ("ER+AEPOG", "AEPOG")])
def test_reference_variant_mapping(variant, reference):
    report = aggregate_reports([evaluate_records(records(variant=variant))], expected_seeds=[0])
    assert report["paper_differences"]["AUROC"]["paper_percent"] == report["paper_reference"]["values"][reference]["AUROC"]


def test_unknown_reference_has_no_invented_paper_scores():
    report = aggregate_reports([evaluate_records(records(variant="ER+Perturb"))], expected_seeds=[0])
    assert report["paper_differences"] == {}


@pytest.mark.parametrize("expected", [[], [0, 0], [-1], [True]])
def test_invalid_expected_seeds(expected):
    with pytest.raises(ValueError, match="Expected seeds"):
        aggregate_reports([], expected_seeds=expected)


def test_aggregation_rejects_duplicate_unexpected_or_incomparable_reports():
    first = evaluate_records(records())
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        aggregate_reports([first, first])
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        aggregate_reports([first], expected_seeds=[1])
    for field, value in (("split", "val"), ("conventions", {}),
                         ("class_balance", {"IND": 1, "OOD": 1}),
                         ("assumptions", ["different"]), ("deviations", ["different"])):
        changed = evaluate_records(records(seed=1))
        changed[field] = value
        with pytest.raises(ValueError, match="matching|mismatch"):
            aggregate_reports([first, changed])
    changed = evaluate_records(records(seed=1, variant="ER+POG"))
    with pytest.raises(ValueError, match="mismatch"):
        aggregate_reports([first, changed])
    changed = deepcopy(first)
    changed["metrics"]["FPR95"] = changed["metrics"].pop("FPR95-IND")
    with pytest.raises(ValueError, match="explicitly oriented"):
        aggregate_reports([changed])
