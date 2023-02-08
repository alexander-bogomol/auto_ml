import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data_preprocessing import build_preprocessor, build_models_dict
from training import train_classifier
from metrics import METRICS

__all__ = ["AutoBinaryClassifier"]

class AutoBinaryClassifier:
    def __init__(self, models):
        # Save dictionary of model names and sklearn estimators
        self.models_dict = build_models_dict(models)

    def train_all_models(self, data, target_column, metric):
        """Make full training pipeline:
            - data and label preprocessing
            - train all models in models list
            - save all fitted models in attributes
            - save models and scores as DataFrame
        """

        models_and_scores = []
        fitted_models_dict = dict()
        classifier_metric = METRICS[metric]

        self.metric = metric

        # Save preprocessing pipeline for data for future transformation
        self.preprocessor = build_preprocessor(data=data, target_column=target_column)
        self.label_encoder = LabelEncoder()

        # Split data on test and train parts
        data_train, data_test, target_train, target_test = train_test_split(
            data.drop(columns=target_column), data[target_column]
        )

        # Preprocess data separately to prevent data leeking
        train_matrix = self.preprocessor.fit_transform(data_train)
        test_matrix = self.preprocessor.transform(data_test)
        target_train_enc = self.label_encoder.fit_transform(target_train)
        target_test_enc = self.label_encoder.transform(target_test)

        # Go through the list and train the models
        for name, model in self.models_dict.items():
            classifier, score = train_classifier(
                model,
                train_matrix,
                test_matrix,
                target_train_enc,
                target_test_enc,
                classifier_metric,
            )
            # Add fitted models to dictionary
            fitted_models_dict[name] = classifier
            # Add scores
            models_and_scores.append([name, score])

        # Save models and scores as DataFrame for users convenience
        self.models_and_scores = pd.DataFrame(
            columns=["Model", metric], data=models_and_scores
        )
        self.fitted_models_dict = fitted_models_dict
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
        return self.fitted_models_dict[best_model_name]

    def best_predict(self, data, inverse_transform=True):
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

        # Make a dictionary that proceeds fi_type into model name from fitted_models_dict
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
            coefficients = self.fitted_models_dict[model].feature_importances_
        # If model is not in fitted_models_dict return KeyError
        except KeyError:
            raise KeyError(f"{model} is not in the list of trained models")
        # Logistic Regression has no `.feature_importances_` attribute, use coefficents
        except AttributeError:
            coefficients = self.fitted_models_dict[model].coef_[0]
        # Get collumns names from ColumnTransformer pipeline
        features = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(
            columns=["Features", "Feature Importance"],
            data=zip(features, coefficients),
        )
        
