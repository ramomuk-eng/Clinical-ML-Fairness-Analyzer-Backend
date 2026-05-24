import numpy as np

def equalized_odds(y_true, y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    tpr_list = []
    fpr_list = []
    group_metrics = {}

    for group in groups:
        mask = sensitive_attr == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        # True positive rate
        tp = np.sum((y_t == 1) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # False positive rate
        fp = np.sum((y_t == 0) & (y_p == 1))
        tn = np.sum((y_t == 0) & (y_p == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        group_metrics[group] = {
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4),
            'count': int(np.sum(mask))
        }

    tpr_sd = round(float(np.std(tpr_list)), 4)
    fpr_sd = round(float(np.std(fpr_list)), 4)

    return {
        'tpr_sd': tpr_sd,
        'fpr_sd': fpr_sd,
        'group_metrics': group_metrics
    }

def classification_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        roc_auc_score, confusion_matrix
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    auroc = round(float(roc_auc_score(y_true, y_prob)), 4)
    npv = round(tn / (tn + fn), 4) if (tn + fn) > 0 else 0

    return {
        'auroc': auroc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'npv': npv
    }