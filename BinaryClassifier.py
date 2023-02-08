from typing import Union, List, Dict, Optional, Callable

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from data_preprocessing import build_preprocessor, build_models_dict
from training import train_classifier
from metrics import *
from models import *

__all__ = [
    "AutoBinaryClassifier",
    "SVM",
    "LOGREG",
    "KNN",
    "DECISIONTREE",
    "NAIVEBAYES",
    "RANDOMFOREST",
    "MODELS",
    "F1",
    "ACCURACY",
    "PRECISION",
    "RECALL",
    "METRICS",
]


class AutoBinaryClassifier:
    """A class that allows you to train several classification models in one line of code and
    choose the best one based on the specified metric. Includes several convenient methods:
    - train_all_models
    - feature_importance
    - best_predict
    - get_best_model
    - models_ranking
    """

    def __init__(self, models: List[str]) -> None:
        # Save dictionary of model names and sklearn estimators
        if len(models) == 0:
            raise ValueError(
                "You've entered an empty list of models. Please input at least one model"
            )
        self.models = build_models_dict(models)
        self.fitted_models = dict()

    def train_all_models(
        self, data: pd.DataFrame, target_column: str, metric: str
    ) -> pd.DataFrame:
        """Make full training pipeline:
        - data and label preprocessing
        - train all models in models list
        - save all fitted models in attributes
        - save models and scores as DataFrame
        """
        models_and_scores = []
        # Get sklearn metric function from dict
        classifier_metric: Callable = METRICS[metric]
        # Save name of metric
        self.metric = metric
        # Save preprocessing pipeline for data for future transformation
        self.preprocessor: ColumnTransformer = build_preprocessor(
            data=data, target_column=target_column
        )
        self.label_encoder = LabelEncoder()

        # Split data on test and train parts
        data_train, data_test, target_train, target_test = train_test_split(
            data.drop(columns=target_column), data[target_column]
        )

        # Preprocess data separately to prevent data leeking
        train_matrix: np.ndarray = self.preprocessor.fit_transform(data_train)
        test_matrix: Union[np.ndarray, pd.DataFrame] = self.preprocessor.transform(
            data_test
        )
        target_train_enc: np.ndarray = self.label_encoder.fit_transform(target_train)
        target_test_enc: np.ndarray = self.label_encoder.transform(target_test)

        # Go through the list and train the models
        for name, model in self.models.items():
            classifier, score = train_classifier(
                model,
                train_matrix,
                test_matrix,
                target_train_enc,
                target_test_enc,
                classifier_metric,
            )
            # Add fitted models to dictionary
            self.fitted_models[name] = classifier
            # Add scores
            models_and_scores.append([name, score])

        # Save models and scores as DataFrame for users convenience
        self.models_and_scores = pd.DataFrame(
            columns=["Model", metric], data=models_and_scores
        )
        return self.models_and_scores

    def models_ranking(self, top=10):
        """Return top n models sorted by score from top to bottom"""
        ranking = (
            self.models_and_scores.sort_values(self.metric, ascending=False)
            .reset_index(drop=True)
            .head(top)
        )
        return ranking

    def get_best_model(self):
        """Return the most scored model"""
        best_model_name = self.models_ranking(top=1).iloc[0, 0]
        return self.fitted_models[best_model_name]

    def best_predict(
        self, data: pd.DataFrame, inverse_transform: Optional[bool] = True
    ):
        """Return predictions from the most scored model,
        producing or not labels decoding
        """
        # Preprocess data
        X = self.preprocessor.transform(data)
        model = self.get_best_model()

        # Check is produce decode labels
        if inverse_transform:
            predictions = model.predict(X)
            return self.label_encoder.inverse_transform(predictions)
        else:
            return model.predict(X)

    def feature_importance(self, fi_type="logreg"):
        """Return feature importance per columns according to feature importance type.
        Available `fi_type` values are 'logreg', 'forest', 'tree'"""

        # Make a dictionary that proceeds fi_type into model name from fitted_models
        available_fi_models = {
            "logreg": "Logistic Regression",
            "forest": "Random Forest",
            "tree": "Decision Tree",
        }

        try:
            # Choose model according to
            model = available_fi_models[fi_type]
        except KeyError:
            raise KeyError(
                "Incorrect fi_type. You have to use one of these values: 'logreg', 'forest', 'tree'"
            )
        try:
            coefficients = self.fitted_models[model].feature_importances_
        # If model is not in fitted_models return KeyError
        except KeyError:
            raise KeyError(f"{model} is not in the list of trained models")
        # Logistic Regression has no `.feature_importances_` attribute, use coefficents
        except AttributeError:
            coefficients = self.fitted_models[model].coef_[0]
        # Get collumns names from ColumnTransformer pipeline
        features = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(
            columns=["Features", "Feature Importance"],
            data=zip(features, coefficients),
        )
