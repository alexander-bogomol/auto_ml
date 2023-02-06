import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.data_preprocessing import build_preprocessor, build_models_dict
from utils.training_models import train_model, METRICS_DICT


class AutoBinaryClassifier:
    def __init__(self, models):
        # Save available imported metrics 
        self.METRICS_DICT = METRICS_DICT
        # Save dictionary of model names and sklearn estimators
        self.models_dict = build_models_dict(models)

    def train_all_models(self, data, target_column, metric):
        models_and_scores = []
        fitted_models_dict = dict()
        classifier_metric = self.METRICS_DICT[metric]

        self.metric = metric
        self.preprocessor = build_preprocessor(data=data, target_column=target_column)
        self.label_encoder = LabelEncoder()

        data_train, data_test, target_train, target_test = train_test_split(
            data.drop(columns=target_column), data[target_column]
        )

        X_train = self.preprocessor.fit_transform(data_train)
        X_test = self.preprocessor.transform(data_test)
        y_train = self.label_encoder.fit_transform(target_train)
        y_test = self.label_encoder.transform(target_test)

        for name, model in self.models_dict.items():
            classifier, score = train_model(
                model, X_train, X_test, y_train, y_test, classifier_metric
            )
            fitted_models_dict[name] = classifier
            models_and_scores.append([name, score])

        self.models_and_scores = pd.DataFrame(
            columns=["Model", metric], data=models_and_scores
        )
        self.fitted_models_dict = fitted_models_dict
        return self.models_and_scores

    def models_ranking(self, top=10):
        ranking = (
            self.models_and_scores.sort_values(self.metric, ascending=False)
            .reset_index(drop=True)
            .head(top)
        )
        return ranking

    def get_best_model(self):
        best_model_name = self.models_ranking(top=1).iloc[0, 0]
        return self.fitted_models_dict[best_model_name]

    def best_predict(self, data, inverse_transform=True):
        X = self.preprocessor.transform(data)
        model = self.get_best_model()
        if inverse_transform:
            predictions = model.predict(X)
            return self.label_encoder.inverse_transform(predictions)
        else:
            return model.predict(X)

    def feature_importance(self, fi_type="logreg"):
        if fi_type == "logreg":
            if "Logistic Regression" in self.fitted_models_dict:
                coefficients = self.fitted_models_dict["Logistic Regression"].coef_[0]
                features = self.preprocessor.get_feature_names_out()
                return pd.DataFrame(
                    columns=["Features", "Correlation"],
                    data=zip(features, coefficients),
                )
            else:
                print("First you need to train the logistic regression model")
        elif fi_type == "tree":
            if "Random Forest" in self.fitted_models_dict:
                coefficients = self.fitted_models_dict[
                    "Random Forest"
                ].feature_importances_
                features = self.preprocessor.get_feature_names_out()
                return pd.DataFrame(
                    columns=["Features", "Feature Importance"],
                    data=zip(features, coefficients),
                )
            else:
                return "First you need to train the random forest model"
        else:
            return "Pass the correct value to the feature importance function"
