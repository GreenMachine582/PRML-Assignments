from __future__ import annotations

import logging
import os

from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from seaborn import heatmap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False)
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

    results = {'accuracy': accuracy_score(y_test, y_pred),
               'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred),
               'f1': f1_score(y_test, y_pred)}

    if show:
        print('Accuracy: %.4f' % results['accuracy'])
        print('Precision: %.4f' % results['precision'])
        print('Recall: %.4f' % results['recall'])
        print('F1: %.4f' % results['f1'])
    return results
