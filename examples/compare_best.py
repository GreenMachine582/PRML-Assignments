from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import machine_learning as ml
from examples import compare_params
from machine_learning import Config, Dataset, Model


def compareEstimator(estimator, dataset, config):
    # TODO: documentation
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{estimator.type_}_{estimator.name}")

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    logging.info("Fitting and predicting")
    estimator.model.fit(X_train, y_train)
    y_pred = np.clip(estimator.model.predict(X_test), 0, None)

    estimator.save()

    estimator.resultAnalysis(y_test, y_pred, plot=False, dataset_name=dataset.name, results_dir=results_dir)
    estimator.plotPrediction(y_test, y_pred, y_train, ylabel="Close ($USD)", dataset_name=dataset.name,
                             results_dir=results_dir)
    estimator.plotImportance(X_train.columns, X_test, y_test, dataset_name=dataset.name, results_dir=results_dir)


def compareClassifier(classifier, dataset, config):
    # TODO: documentation
    results_dir = ml.utils.makePath(config.dir_, config.results_folder, f"{classifier.type_}_{classifier.name}")

    dataset.df.drop('diff', axis=1, inplace=True)  # same feature as binary encoded target
    dataset.apply(ml.binaryEncode, dataset.target)

    X_train, X_test, y_train, y_test = dataset.split(shuffle=False)

    logging.info("Fitting and predicting")
    classifier.model.fit(X_train, y_train)
    y_pred = classifier.model.predict(X_test)

    classifier.save()

    classifier.resultAnalysis(y_test, y_pred, plot=False, dataset_name=f"{dataset.name} Recorded Best")
    classifier.plotPrediction(y_test, y_pred, y_train, dataset_name=f"{dataset.name} Recorded Best")
    classifier.plotImportance(X_train.columns, X_test, y_test, dataset_name=dataset.name, results_dir=results_dir)


def compareBest(dataset: Dataset, config: Config) -> None:
    """

    :param dataset: The loaded and processed dataset, should be a Dataset
    :param config:
    :return: None
    """
    # TODO: documentation

    while True:
        print(f"""
        0 - Back
        Estimators:
            1 - Gradient Boosting Regressor
            2 - Random Forest Regressor
            3 - Decision Tree Regressor
        Classifiers:
            4 - Ridge Classifier
        """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        model_config = None
        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                model_config = compare_params.getGradientBoostingRegressor(config.random_state)
            elif choice == 2:
                model_config = compare_params.getRandomForestRegressor(config.random_state)
            elif choice == 3:
                model_config = compare_params.getDecisionTreeRegressor(config.random_state)
            elif choice == 4:
                model_config = compare_params.getRidgeClassifier(config.random_state)
            else:
                print("\nPlease enter a valid choice!")

        if model_config is not None:
            model = Model(config.model, **model_config)
            model.base = Pipeline([('scaler', StandardScaler()), (model.name, model.base)])
            model.createModel()
            if model.type_ == 'estimator':
                compareEstimator(model, deepcopy(dataset), config)
            elif model.type_ == 'classifier':
                compareClassifier(model, deepcopy(dataset), config)
