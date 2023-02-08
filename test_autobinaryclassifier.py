import pytest
from BinaryClassifier import *


@pytest.fixture
def build_AutoBinaryClassifier_instance():
    return AutoBinaryClassifier(
        [SVM, DECISIONTREE, RANDOMFOREST, LOGREG, KNN, NAIVEBAYES]
    )

@pytest.mark.parametrize(
    "fi_type, error",
    [
        ("logreg", "Logistic Regression is not in the list of trained models"),
        ("tree", "Decision Tree is not in the list of trained models"),
        ("forest", "Random Forest is not in the list of trained models"),
        ("123", "Incorrect fi_type. You have to use one of these values: 'logreg', 'forest', 'tree'"),
    ],
)
def test_abc_feature_importance(build_AutoBinaryClassifier_object, fi_type, error):
    with pytest.raises(KeyError) as exc_info:
        build_AutoBinaryClassifier_object.feature_importance(fi_type)
    assert exc_info.value.args[0] == error

def test_abc__init__instance():
    with pytest.raises(ValueError) as exc_info:
        AutoBinaryClassifier([])
    assert exc_info.value.args[0] == "You've entered an empty list of models. Please input at least one model"
