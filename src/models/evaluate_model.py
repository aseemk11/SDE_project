# evaluate_model.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model_function(y_true, y_pred, y_prob=None):
    """
    Evaluates the performance of the model using common classification metrics.
    
    Args:
    - y_true (list or array): True labels.
    - y_pred (list or array): Predicted labels by the model.
    - y_prob (list or array, optional): Predicted probabilities for AUC calculation.
    
    Returns:
    - dict: Dictionary containing evaluation metrics.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_prob)
    
    return results
