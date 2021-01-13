from sklearn.metrics import confusion_matrix

def get_basic_metrics(actuals, predictions):
    """
    Get per class true positives, true negatives,
    false positives, and false negatives.

    :param actuals: The actual labels in the data (a.k.a y)
    :param predictions: The labels predicted (a.k.a y_hat)

    :returns Tuple: A tuple with per-class TP, FP, TN, FN.
    """
    conf_mat = confusion_matrix(
        actuals,
        predictions,
        labels=labs
    )

    # Per-label TP, FP, TN, FN
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)

    return TP, FP, TN, FN


def get_metrics(actuals, predictions):
    
