from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

__all__ = ["train_model", "METRICS_DICT", "MODELS_DICT"]

# Dictionary of available metrics. auto_ml name to sclearn functions.
METRICS_DICT = {
    "F1": f1_score,
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
}
# Dictionary of available models. auto_ml name to sclearn estimators.
MODELS_DICT = {
    "Support Vector Machines": SVC,
    "Logistic Regression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Naive Bayes": GaussianNB,
    "Random Forest": RandomForestClassifier,
}


def train_model(model, X_train, X_test, target_train, target_test, metric):
    estimator = model()
    estimator.fit(X_train, target_train)
    predictions = estimator.predict(X_test)
    score = metric(target_test, predictions)
    return estimator, score
