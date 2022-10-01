from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn import ensemble, neighbors, neural_network, svm, tree
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import examples
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def plotPredictions(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                    y_pred: ndarray) -> None:
    """
    Plots the BTC daily Close and predictions.

    :param X_train: Training independent features, should be a DataFrame
    :param X_test: Testing independent features, should be a DataFrame
    :param y_train: Training independent features, should be a DataFrame
    :param y_test: Testing dependent features, should be a DataFrame
    :param y_pred: Predicted dependent variables, should be a ndarray
    :return: None
    """
    # plots a line graph of BTC True and Predicted Close vs Date
    plt.figure()
    plt.plot(X_train.index, y_train, color='blue', label='Train')
    plt.plot(X_test.index, y_test, color='green', label='Test')
    plt.plot(X_test.index, y_pred, color='red', label='Predictions')
    plt.title('BTC Close Vs Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()

    plt.figure()
    plt.plot(X_test.index, y_test, color='green', label='Test')
    plt.plot(X_test.index, y_pred, color='red', label='Predictions')
    plt.title('BTC Close Vs Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()
    plt.show()


# def main(dir_=local_dir):
#     config = ml.Config(dir_, 'BTC-USD')
#
#     dataset = ml.Dataset(config.dataset)
#     if not dataset.load():
#         return
#
#     dataset = examples.processDataset(dataset)
#
#     X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)
#     X_train.sort_index(inplace=True)
#     X_test.sort_index(inplace=True)
#     y_train.sort_index(inplace=True)
#     y_test.sort_index(inplace=True)
#
#     # compareModels(X_train, y_train, cv=TimeSeriesSplit(10), random_state=config.random_state)
#
#     pipeline = Pipeline([('scaler', StandardScaler()),
#                          ('EGBR', ensemble.GradientBoostingRegressor())])
#     param_grid = {'EGBR__learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
#                   'EGBR__max_depth': [3, 10, 20, 25, 50],
#                   'EGBR__n_estimators': [500, 800, 1000, 1200],
#                   'EGBR__subsample': [0.10, 0.5, 0.7, 1.0],
#                   }
#
#     defaults = {'n_jobs': -1, 'cv': TimeSeriesSplit(10), 'verbose': 2, 'return_train_score': True}
#     cv_results = GridSearchCV(pipeline, param_grid, **defaults)
#     cv_results.fit(X_train, y_train)
#     print('The best estimator:', cv_results.best_estimator_)
#     print('The best score:', cv_results.best_score_)
#     print('The best params:', cv_results.best_params_)
#
#     model = ml.Model(config.model, model=cv_results.best_estimator_)
#     model.fit(X_train, y_train)
#     model.save()
#
#     y_pred = model.predict(X_test)
#     ml.resultAnalysis(y_test, y_pred)
#     plotPredictions(X_train, X_test, y_train, y_test, y_pred)
#
#     logging.info(f"Completed")
#     return


def main(dir_: str = local_dir) -> None:
    """
    Gives the user a choice between tasks or datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """

    config = ml.Config(dir_, 'BTC-USD')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")

    run = True
    while run:
        print("""
        0 - Back
        1 - Process Dataset (Includes EDA)
        2 - Compare Estimators
        2 - Compare Classifiers
        3 - Find Best Params (TBA)
        4 - Plot Best Predictions (TBA)
        """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                examples.process.main(dataset)
                return
            elif choice == 2:
                examples.compare_models.compareEstimators(dataset, config.random_state)
                return
            elif choice == 3:
                examples.compare_models.compareClassifiers(dataset, config.random_state)
                return
            elif choice == 4:
                pass
            elif choice == 5:
                pass
            else:
                print("\nPlease enter a valid choice!")


if __name__ == '__main__':
    main()
