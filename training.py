def train_classifier(model, X_train, X_test, target_train, target_test, metric):
    """Return fitted model and its score"""
    estimator = model()
    estimator.fit(X_train, target_train)
    predictions = estimator.predict(X_test)
    score = metric(target_test, predictions)
    return estimator, score
