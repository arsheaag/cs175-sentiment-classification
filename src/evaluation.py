from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred, y_prob):
    """
    Compute evaluation metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    report = classification_report(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "roc_auc": auc,
        "classification_report": report
    }
