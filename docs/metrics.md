# Evaluation metrics

## Input and units

Use `oos_detect.reproduce.evaluation` to make reports from saved probabilities.
This module does not load a model. It does not execute inference.
Run inference on CUDA before you use this module.

The integrated `evaluate` CLI stage loads the selected classifier, runs CUDA
inference, and writes these reports. The `aggregate` stage combines explicit
summary files. See [reproduction commands](reproduction-commands.md) for commands
and ablations.

Each example must have probabilities for IND intents only. Do not add an OOS intent.
The probabilities must be finite, between zero and one, and sum to one.
The sum tolerance is `1e-6`. An intent tie selects the lowest intent ID.
OOD examples must have `target=None`.

The confidence is `c = max(probabilities)`. The OOD score is `1-c`.
Each evaluation needs both IND and OOD examples. Missing classes cause an error.
Duplicate source IDs, mixed splits, and inconsistent provenance also cause an error.

All computed metrics use fractions, not percentages. Higher AUROC and AUPR values
are better. Lower FPR values are better.

## Metric definitions

| Field | Definition |
| --- | --- |
| `IND-accuracy` | Correct IND intent predictions divided by the IND example count. OOD examples do not enter this calculation. |
| `AUROC` | ROC area with OOD positive and score `1-c`. This equals IND-positive ROC area with score `c`. |
| `AUPR-IND` | Non-interpolated average precision with IND positive and score `c`. |
| `AUPR-OOD` | Non-interpolated average precision with OOD positive and score `1-c`. |
| `FPR95-IND` | OOD acceptance rate at the first IND-positive ROC point with TPR at least 0.95. |
| `FPR90-IND` | OOD acceptance rate at the first IND-positive ROC point with TPR at least 0.90. |
| `FPR95-OOD` | IND rejection rate at the first OOD-positive ROC point with TPR at least 0.95. |
| `FPR90-OOD` | IND rejection rate at the first OOD-positive ROC point with TPR at least 0.90. |

The suffix names the positive class whose TPR is the target.
There is no unsuffixed reproduction FPR metric.
The paper reference has its original field names in a separate `paper_reference` object.

Average precision is `sum((recall_next - recall_previous) * precision_next)`.
It is not trapezoidal precision-recall area.
The implementation uses scikit-learn
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
and [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

## ROC thresholds and ties

For each orientation, predict positive when `score >= threshold`.
Use [`roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
with `pos_label=1` and `drop_intermediate=False`.
Process thresholds from highest to lowest. Include all examples with an equal score together.
Select the first observed point whose TPR meets the target. Do not interpolate.
The actual TPR can exceed the target when a score tie crosses it.

For example, let 18 IND examples have confidence 0.9. Let two IND examples and
one OOD example have confidence 0.8. The IND-positive 90% point has FPR zero.
The IND-positive 95% point includes the whole 0.8 group. Its actual TPR and FPR
are both one. Do not split the group to produce an artificial 95% TPR point.

The `roc_operating_points` object records the positive class, score, comparison,
target TPR, actual TPR, FPR, and threshold for each named metric.
Its kind is `descriptive-split-ROC-not-deployment`.
The local `fpr` field is scoped by its named metric and `positive_class`.

## Validation and checkpoint provenance

Select the classifier checkpoint with IND-validation accuracy only.
The default rule selects the earliest epoch when accuracies are equal.
Fix this selection before test inference. Retain the selected checkpoint file and its SHA-256 digest.

Create `EvaluationProvenance` with these fields:

| Field | Required value |
| --- | --- |
| `seed` | Nonnegative integer run seed. |
| `variant` | Experiment name, for example `MSP` or `ER+POG`. |
| `data_sha256` | Lowercase SHA-256 digest of the prepared data artifact. |
| `vocabulary_sha256` | Lowercase SHA-256 digest of the classifier vocabulary artifact. |
| `selected_checkpoint` | Selected checkpoint path or stable artifact name. |
| `selected_checkpoint_sha256` | Lowercase SHA-256 digest of the selected checkpoint bytes. |
| `selection_split` | `val`; this is the default and only accepted value. |
| `selection_criterion` | Nonempty description; the default is best IND-validation accuracy with earliest-epoch ties. |

The caller must compute these digests from the actual artifacts.
The report module checks identities. It cannot prove that training obeyed the declared selection rule.
Keep the training manifest and selection logs for that audit.

To select an optional deployment threshold, call `calibrate_threshold` on validation records only.
The result has kind `validation-calibrated-deployment` and a validation-record digest.
Pass this result to `evaluate_records` as `deployment_threshold`.
Its checkpoint, seed, variant, data, vocabulary, and selection provenance must match.
The test report applies the fixed threshold without adjustment.
Its actual test TPR need not meet the validation target.

Do not use a descriptive test ROC threshold as a deployment threshold.
The two objects have different kinds and separate report locations.
The API rejects test records for calibration and rejects mismatched threshold provenance.

## Report API and saved files

These functions accept precomputed Python probability rows. For tensor output,
first detach it and convert it to CPU lists after CUDA inference.

| API | Result |
| --- | --- |
| `prediction_records(rows, probabilities, *, provenance, split)` | Tuple of validated `PredictionRecord` objects. Each source row needs `id`, `partition` or `split`, `is_ood`, and `target`. |
| `calibrate_threshold(records, *, positive="OOD", target_tpr=0.95)` | Validation-only threshold dictionary. |
| `evaluate_records(records, *, deployment_threshold=None, assumptions=(), deviations=())` | Per-run report dictionary. Does not write files. |
| `save_evaluation(output_dir, records, **kwargs)` | Creates a new directory, writes predictions and summary, and returns the summary. Accepts evaluation keyword arguments. |
| `aggregate_reports(reports, *, expected_seeds=range(10))` | Aggregate dictionary for one comparable variant. Does not write files. |

`PredictionRecord` fields are `id`, `split`, `is_ood`, `target`, `prediction`,
`probabilities`, and `provenance`. Its `confidence` property retains maximum softmax.
Its `to_dict()` method also adds `confidence` and `ood_score`.

`save_evaluation` writes:

- `predictions.jsonl`: one record per source ID, with probabilities, confidence,
  split, OOD status, intent prediction, target, and full provenance.
- `summary.json`: metrics, class balance, conventions, provenance, assumptions,
  deviations, optional threshold application, and the prediction-file SHA-256 digest.

An existing output directory causes an error. It is not overwritten.
The prediction digest identifies the exact JSONL bytes.

## Seed aggregation

Declare the expected seed set. The default is seeds 0 through 9.
Keep each run report under `per_seed`. Compute each metric's mean from completed
seeds only. Compute sample standard deviation with denominator `n-1`.
For one seed, `sample_sd` is `null`. For no seeds, both statistics are `null`.

An aggregate is `complete` only when all expected seeds are present.
Missing seeds remain explicit. Duplicate and unexpected seeds cause an error.
Aggregation requires test reports with the same variant, data and vocabulary
digests, selection rule, class balance, conventions, assumptions, and deviations.
Selected checkpoint paths and digests can differ between seeds.
If preparation differs between seeds, provide separate aggregates or first establish
a shared comparable prepared artifact. Do not replace digests to bypass a mismatch.

This is a shortened expected output for one analytic seed. It is not a measured paper run:

```json
{
  "schema_version": 1,
  "variant": "MSP",
  "status": "incomplete",
  "expected_seeds": [0, 1],
  "completed_seeds": [0],
  "missing_seeds": [1],
  "statistics": {
    "AUROC": {"n": 1, "mean": 1.0, "sample_sd": null}
  },
  "assumptions": [
    "Paper classifier vocabulary uses declared train/validation/test text, without held-out labels (transductive).",
    "tiny analytic fixture"
  ],
  "deviations": ["two intents, not paper profile"],
  "numerical_parity_gate": false
}
```

The full report also has all metrics, per-seed reports, comparison identity,
paper references, and provisional differences.
The artifact test checks that assumptions, deviations, and missing seeds survive report creation.

## Paper references and limitations

The change records these OSQ/CLINC references from paper `1909.03862v4`.
All values in this table are percentages.

| Variant | AUROC | AUPR | FPR95 | FPR90 | IND accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSP | 92.86 | 98.24 | 39.72 | 21.76 | 92.61 |
| POG | 95.41 | 98.94 | 25.00 | 10.10 | 93.31 |
| AEPOG | 95.83 | 99.05 | 23.70 | 9.50 | 93.32 |

The paper designates AUPR as OOD-positive. Its FPR positive class is ambiguous.
Keep both reproduction orientations. Do not select the orientation that is closest to the paper.
The `AUPR-IND` difference is explicitly an orientation mismatch, not an equivalent comparison.
Both FPR differences state that the paper orientation is ambiguous.
Differences use `100 * reproduction_mean - paper_percent`, in percentage points.
There is no numerical-parity pass/fail gate. Seed completeness does not imply paper parity.
Variants without a listed reference get no invented paper comparison.

AUPR depends on class balance. The full test split has 4,500 IND and 1,000 OOD
examples. If all scores are tied, AUROC is 0.5 and average precision equals the
positive-class fraction. AUPR-IND and AUPR-OOD thus differ even with the same scores.

The paper vocabulary profile uses validation and test text to build the classifier
vocabulary. This is transductive text use, not held-out label use.
The generator vocabulary uses IND training text only.
Disclose a train-only alternative in `deviations`; do not call it the same profile.
Also record engineering defaults, embedding choices, and non-paper smoke settings.

## Verification

`tests/reproduction/test_evaluation.py` has analytic fixtures for perfect and
reversed separation, imbalanced rankings, tied scores, threshold target crossings,
missing classes, artifact identity, validation-only calibration, and seed aggregation.
These analytic checks do not need model execution or data downloads. The locked
Colab T4 runtime passed all 258 reproduction tests and the seven-case CLI smoke.
See the [acceptance report](acceptance-report.md). Generator CLI resume passed
from POG boundary 4 to boundary 20 in a fresh process. Classifier epoch resume
passed API tests; a separate classifier CLI resume was not run.
Private paper datasets, full campaigns, and paper scores remain unverified.
Passing tests does not establish paper numerical parity.
