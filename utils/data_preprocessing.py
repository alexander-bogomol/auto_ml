from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils.training_models import MODELS_DICT

# Add more models in future

__all__ = ["build_preprocessor", "build_models_dict"]


def build_preprocessor(data, target_column):
    """
    """
    data = data.drop(columns=target_column)

    categorical_columns = list(data.select_dtypes(include=["object"]).columns.values)
    numerical_columns = list(
        data.select_dtypes(include=["float", "int"]).columns.values
    )

    data_preprocessor = ColumnTransformer(
        [
            ("numerical", StandardScaler(), numerical_columns),
            ("categorical", OneHotEncoder(), categorical_columns),
        ]
    )

    return data_preprocessor

def build_models_dict(models: list):
    models_dict = dict()
    for model in models:
        models_dict[model] = MODELS_DICT[model]
    return models_dict
