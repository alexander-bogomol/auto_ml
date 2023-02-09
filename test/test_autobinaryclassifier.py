import pytest
import pandas as pd
import numpy as np
from src.BinaryClassifier import *


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
def test_abc_feature_importance(build_AutoBinaryClassifier_instance, fi_type, error):
    with pytest.raises(KeyError) as exc_info:
        build_AutoBinaryClassifier_instance.feature_importance(fi_type)
    assert exc_info.value.args[0] == error

@pytest.mark.parametrize(
    "models_list, error",
    [
        ([], "You've entered an empty list of models. Please input at least one model"),
        ("Linear Regression", "Please enter the list of available models"),
        (123, "Please enter the list of available models"),
    ],
)
def test_abc__init__assertions(models_list, error):
    with pytest.raises(AssertionError) as exc_info:
        AutoBinaryClassifier(models_list)
    assert exc_info.value.args[0] == error

def test_abc__train_all_models_data_assertion(build_AutoBinaryClassifier_instance):
    with pytest.raises(AssertionError) as exc_info:
        build_AutoBinaryClassifier_instance.train_all_models([1, 2, 3], target_column="Survived", metric="F1")
    assert exc_info.value.args[0] == "Only pandas DataFrame are currently supported"

def test_abc__train_all_models_functional(build_AutoBinaryClassifier_instance):
    data = pd.read_csv("dataset/X.csv", dtype={"Sex": "object", "Pclass": "object"})
    table = build_AutoBinaryClassifier_instance.train_all_models(data=data, target_column="Survived", metric="F1")
    # Assert output is DataFrame
    assert isinstance(table, pd.DataFrame)
    # Assert 'Model' column contain correct values
    assert np.array_equal(table['Model'].values, np.array(['Support Vector Machines', 'Decision Tree', 'Random Forest',
       'Logistic Regression', 'KNN', 'Naive Bayes']))
    # Assert 'F1' column contain array of floats
    assert table["F1"].values.dtype.kind == 'f'
...
