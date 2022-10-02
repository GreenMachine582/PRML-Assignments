from __future__ import annotations

import logging
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import ensemble, neighbors, neural_network, svm, tree, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import examples
from machine_learning import Dataset

# Constants
local_dir = os.path.dirname(__file__)


def compareModels(models: list, X_train: DataFrame, y_train: DataFrame) -> dict:
    # TODO: Fix documentation
    results = {}
    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=TimeSeriesSplit(10), n_jobs=-1)
        results[name] = cv_results

        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return results


def plotPredictions(X_train: DataFrame, X_test: DataFrame, y_train: DataFrame, y_test: DataFrame,
                    predictions: list) -> None:
    """
    Plots the BTC daily Close and predictions.

    :param X_train: Training independent features, should be a DataFrame
    :param X_test: Testing independent features, should be a DataFrame
    :param y_train: Training independent features, should be a DataFrame
    :param y_test: Testing dependent features, should be a DataFrame
    :param predictions: Predicted dependent variables, should be a list[tuple[str: ndarray]]
    :return: None
    """
    # TODO: Fix documentation
    # plots a line graph of BTC True and Predicted Close vs Date
    plt.figure()
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(predictions) + 2)))
    c = next(colour)
    plt.plot(X_train.index, y_train, c=c, label='Train')
    c = next(colour)
    plt.plot(X_test.index, y_test, c=c, label='Test')
    for name, y_pred in predictions:
        c = next(colour)
        plt.plot(X_test.index, y_pred, c=c, label=f"{name} Predictions")
    plt.title('BTC Predicted Close')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()

    plt.figure()
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(predictions) + 2)))
    c = next(colour)
    plt.plot(X_test.index, y_test, c=c, label='Test')
    for name, y_pred in predictions:
        c = next(colour)
        plt.plot(X_test.index, y_pred, c=c, label=f"{name} Predictions")
    plt.title('BTC Predicted Close (Closeup)')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()
    plt.show()


def compareEstimators(dataset: Dataset, random_state: int = None) -> None:
    # TODO: Fix documentation
    dataset = examples.processDataset(dataset)

    models = [('EGBR', ensemble.GradientBoostingRegressor(random_state=random_state)),
              ('ERFR', ensemble.RandomForestRegressor(random_state=random_state)),
              ('NKNR', neighbors.KNeighborsRegressor()),
              ('NNMLPR', neural_network.MLPRegressor(random_state=random_state)),
              ('SSVR', svm.SVR()),
              ('TDTR', tree.DecisionTreeRegressor(random_state=random_state)),
              ('TETR', tree.ExtraTreeRegressor(random_state=random_state))]

    X_train, X_test, y_train, y_test = dataset.split(random_state=random_state, shuffle=False)

    scores = compareModels(models, X_train, y_train)

    plt.figure()
    plt.boxplot(scores.values(), labels=scores.keys())
    plt.title('Algorithm Comparison')
    plt.show()

    del scores['NKNR']
    del scores['NNMLPR']
    del scores['SSVR']

    plt.figure()
    plt.boxplot(scores.values(), labels=scores.keys())
    plt.title('Algorithm Comparison')
    plt.show()

    predictions = []
    for name, estimator in models:
        if name in scores:
            estimator.fit(X_train, y_train)
            predictions.append((name, estimator.predict(X_test)))

    plotPredictions(X_train, X_test, y_train, y_test, predictions)


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


def plotClassifications(y_test: DataFrame, names: list, predictions: list):
    # Heatmaps
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    df_cm = DataFrame(confusion_matrix(y_test, predictions[0]))
    sns.heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False, ax=ax1)
    ax1.set_ylabel(f"{names[0]}", fontsize=14)
    df_cm = DataFrame(confusion_matrix(y_test, predictions[1]))
    sns.heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False, ax=ax2)
    ax2.set_ylabel(f"{names[1]}", fontsize=14)
    df_cm = DataFrame(confusion_matrix(y_test, predictions[2]))
    sns.heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False, ax=ax3)
    ax3.set_ylabel(f"{names[2]}", fontsize=14)
    df_cm = DataFrame(confusion_matrix(y_test, predictions[3]))
    sns.heatmap(df_cm, square=True, annot=True, fmt='d', cbar=False, ax=ax4)
    ax4.set_ylabel(f"{names[3]}", fontsize=14)
    plt.suptitle("BTC Close Classification Predictions - Confusion Matrices")
    plt.show()


def compareClassifiers(dataset: Dataset, random_state: int = None) -> None:
    # TODO: Fix documentation
    dataset = examples.processDataset(dataset)
    dataset.apply(convertToCategorical)

    X_train, X_test, y_train, y_test = dataset.split(random_state=random_state, shuffle=False)

    models = [('EGBC', ensemble.GradientBoostingClassifier(random_state=random_state)),
              ('ERFC', ensemble.RandomForestClassifier(random_state=random_state)),
              ('LMLR', linear_model.LogisticRegression(random_state=random_state)),
              ('LMRC', linear_model.RidgeClassifier(random_state=random_state)),
              ('LMSGDC', linear_model.SGDClassifier(random_state=random_state)),
              ('NKNC', neighbors.KNeighborsClassifier()),
              ('NNC', neighbors.NearestCentroid()),
              ('NNMLPC', neural_network.MLPClassifier(random_state=random_state)),
              ('SSVC', svm.SVC()),
              ('TDTC', tree.DecisionTreeClassifier(random_state=random_state))]

    scores = compareModels(models, X_train, y_train)

    plt.figure()
    plt.boxplot(scores.values(), labels=scores.keys())
    plt.title('Algorithm Comparison')
    plt.show()

    del scores['LMLR']
    del scores['LMSGDC']
    del scores['NKNC']
    del scores['NNC']
    del scores['NNMLPC']
    del scores['SSVC']

    plt.figure()
    plt.boxplot(scores.values(), labels=scores.keys())
    plt.title('Algorithm Comparison')
    plt.show()

    predictions = {}
    for name, classifier in models:
        if name in scores:
            classifier.fit(X_train, y_train)
            predictions[name] = classifier.predict(X_test)

    plotClassifications(y_test, list(predictions.keys()), list(predictions.values()))
