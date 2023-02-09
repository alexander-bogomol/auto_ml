from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

__all__ = ["F1", "ACCURACY", "PRECISION", "RECALL", "METRICS"]

# For ease of import for users, map variables to dictionary keys with sklearn classes
F1 = "F1"
ACCURACY = "Accuracy"
PRECISION = "Precision"
RECALL = "Recall"

# Dictionary of available scikit-learn metrics.
METRICS = {
    "F1": f1_score,
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
}
