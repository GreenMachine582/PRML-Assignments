from __future__ import annotations

import logging
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import explained_variance_score, mean_squared_log_error, r2_score, mean_absolute_error, \
    mean_squared_error, confusion_matrix

# Constants
local_dir = os.path.dirname(__file__)


def convertToCategorical(df: DataFrame) -> DataFrame:
    """
    Encodes the BTC Target into a binary category where,
    0 represents a fall in price from previous instance
    1 represents a rise in price from previous instance.

    :param df: The BTC dataset, should be a DataFrame
    :return: df - DataFrame
    """
    df['Close'] = [0 if df['diff'][i] < 0 else 1 for i in range(len(df['Close']))]

    # Checks for class imbalance
    if len(df['Close'] == 0) != len(df['Close'] == 1):
        logging.warning("Class imbalance is present")

    df['Close'] = df['Close'].astype("category")
    return df


def plotClassifications(y_test: DataFrame, name: str, y_preds: ndarray) -> None:
    """
    Plots the predictions in a confusion matrix format.

    :param y_test: Testing dependent variables, should be a DataFrame
    :param name: Classifier's name, should be a str
    :param y_preds: Classifier predictions, should be a ndarray
    :return: None
    """
    plt.figure()
    df_cm = DataFrame(confusion_matrix(y_test, y_preds))
    sns.heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False)
    plt.suptitle(f"BTC Close Classification Predictions - {name} - Confusion Matrix")
    plt.show()


def resultAnalysis(y_test: DataFrame, y_pred: ndarray, show: bool = True) -> dict:
    """
    Calculates and displays the result analysis for classifiers.

    :param y_test: Testing dependent variables, should be a DataFrame
    :param y_pred: Model predictions, should be a ndarray
    :param show: Whether to show the results, should be a bool
    :return: results - dict[str: float]
    """
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
