from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from typing import Union

__all__ = [
    "SVM",
    "LOGREG",
    "KNN",
    "DECISIONTREE",
    "NAIVEBAYES",
    "RANDOMFOREST",
    "MODELS",
    "TypeClassifier"
]

# For ease of import for users, map variables to dictionary keys with sklearn classes
SVM = "Support Vector Machines"
LOGREG = "Logistic Regression"
KNN = "KNN"
DECISIONTREE = "Decision Tree"
NAIVEBAYES = "Naive Bayes"
RANDOMFOREST = "Random Forest"

# Dictionary of available models. auto_ml name to sclearn estimators.
MODELS = {
    "Support Vector Machines": SVC,
    "Logistic Regression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Naive Bayes": GaussianNB,
    "Random Forest": RandomForestClassifier,
}

TypeClassifier = Union[
    SVC,
    LogisticRegression,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    GaussianNB,
    RandomForestClassifier,
]
