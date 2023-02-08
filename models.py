from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

__all__ = ["SVM", "LOGREG", "KNN", "DECISIONTREE", "NAIVEBAYES", "RANDOMFOREST", "MODELS"]

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