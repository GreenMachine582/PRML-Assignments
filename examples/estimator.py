from __future__ import annotations

import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import explained_variance_score, mean_squared_log_error, r2_score, mean_absolute_error, mean_squared_error

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
    # plots a line graph of BTC True and Predicted Close
    plt.figure()
    plt.plot(X_train.index, y_train, c='b', label='Train')
    plt.plot(X_test.index, y_test, c='r', label='Test')
    plt.plot(X_test.index, y_pred, c='g', label=f"Predictions")
    plt.title('BTC Predicted Close')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()

    plt.figure()
    plt.plot(X_test.index, y_test, c='r', label='Test')
    plt.plot(X_test.index, y_pred, c='g', label=f"Predictions")
    plt.title('BTC Predicted Close (Closeup)')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()
    plt.show()


def resultAnalysis(y_test: DataFrame, y_pred: ndarray, show: bool = True) -> dict:
    """
    Calculates and displays the result analysis for estimators.

    :param y_test: Testing dependent variables, should be a DataFrame
    :param y_pred: Model predictions, should be a ndarray
    :param show: Whether to show the results, should be a bool
    :return: results - dict[str: float]
    """
    # TODO: Separate classified and estimated predictions.
    logging.info("Analysing results")

    results = {'explained_variance': explained_variance_score(y_test, y_pred),
               'mean_squared_log_error': mean_squared_log_error(y_test, y_pred),
               'r2': r2_score(y_test, y_pred),
               'mae': mean_absolute_error(y_test, y_pred),
               'mse': mean_squared_error(y_test, y_pred)}
    results['rmse'] = np.sqrt(results['mse'])

    if show:
        print('explained_variance: %.4f' % results['explained_variance'])
        print('mean_squared_log_error: %.4f' % results['mean_squared_log_error'])
        print('r2: %.4f' % results['r2'])
        print('MAE: %.4f' % results['mae'])
        print('MSE: %.4f' % results['mse'])
        print('RMSE: %.4f' % np.sqrt(results['rmse']))
    return results
