import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def get_basic_metrics(actuals, predictions, labels):
    """
    Get per class true positives, true negatives,
    false positives, and false negatives.

    :param actuals: The actual labels in the data (a.k.a y)
    :param predictions: The labels predicted (a.k.a y_hat)
    :param labels: Set of all labels

    :returns Tuple: A tuple with per-class TP, FP, TN, FN.
    """
    conf_mat = confusion_matrix(
        actuals,
        predictions,
        labels=labels
    )

    # Per-label TP, FP, TN, FN
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)

    return TP, FP, TN, FN


def get_metrics(actuals, predictions, labels):
    """
    Get per class true positives, true negatives,
    false positives, and false negatives.

    :param actuals: The actual labels in the data (a.k.a y)
    :param predictions: The labels predicted (a.k.a y_hat)
    :param labels: Set of all labels

    :returns DataFrame: pandas df with metrics.
    """
    TP, FP, TN, FN = get_basic_metrics(actuals, predictions, labels)

    # Per-label accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    # positive predictive value, prec
    P = TP / (TP + FP)

    # TPR, recall, sensitivity
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    metrics_df = pd.DataFrame({
        "labels": labels,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Precision": P,
        "Recall": R,
        "F-Score": F1,
        "Accuracy": ACC
    })

    return metrics_df.set_index("labels")
