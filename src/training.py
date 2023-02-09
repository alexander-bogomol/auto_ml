from src.models import TypeClassifier
from scipy.sparse import csr_matrix
import numpy.typing as npt
from typing import Tuple, Callable, Union


def train_classifier(
    model: TypeClassifier,
    data_train: Union[npt.ArrayLike, csr_matrix],
    data_test: Union[npt.ArrayLike, csr_matrix],
    target_train: Union[npt.ArrayLike, csr_matrix, None],
    target_test: Union[npt.ArrayLike, csr_matrix, None],
    metric: Callable,
) -> Tuple[TypeClassifier, float]:
    """Return fitted model and its score"""
    estimator = model()  # type: ignore
    estimator.fit(data_train, target_train)
    predictions = estimator.predict(data_test)
    score = metric(target_test, predictions)
    return estimator, score
