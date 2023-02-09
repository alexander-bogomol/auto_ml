from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.models import MODELS, TypeClassifier
import pandas as pd
from typing import List, Dict

__all__ = ["build_preprocessor", "build_models_dict"]


def build_preprocessor(data: pd.DataFrame, target_column: str) -> ColumnTransformer:
    """Make the pipline for data preprocessing"""
    # Drop target column
    data = data.drop(columns=target_column)
    # Create lists of categorical and numerical columns
    categorical_columns = list(
        data.select_dtypes(include=["object"]).columns.values
    )
    numerical_columns = list(
        data.select_dtypes(include=["float", "int"]).columns.values
    )
    # Build the pipline
    data_preprocessor = ColumnTransformer(
        [
            ("numerical", StandardScaler(), numerical_columns),
            ("categorical", OneHotEncoder(), categorical_columns),
        ]
    )

    return data_preprocessor


def build_models_dict(models: List[str]) -> Dict[str, TypeClassifier]:
    """Create dict of models and estimators, checking is estimator available"""
    models_dict = dict()
    for model in models:
        try:
            models_dict[model] = MODELS[model]
        except KeyError:
            raise KeyError(f"{model} is not in the available models list!")
    return models_dict
