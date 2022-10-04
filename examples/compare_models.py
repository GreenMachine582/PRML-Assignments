from __future__ import annotations

import logging
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import ensemble, neighbors, neural_network, svm, tree, linear_model, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_validate

import examples
from machine_learning import Dataset

# Constants
local_dir = os.path.dirname(__file__)


def compareModels(models: dict, X_train: DataFrame, y_train: DataFrame) -> dict:
    # TODO: Fix documentation
    results = {}
    for name in models:
        cv_results = cross_validate(models[name], X_train, y_train, cv=TimeSeriesSplit(10), n_jobs=-1)
        cv_results['model'] = models[name]
        results[name] = cv_results

        print('%s: %f (%f)' % (name, cv_results['test_score'].mean(), cv_results['test_score'].std()))
    return results


def plotEstimatorResultAnalysis(y_test, predictions):
    results = {'names': [], 'explained_variance': [], 'mean_squared_log_error': [],
               'r2': [], 'mae': [], 'mse': [], 'rmse': []}
    for name, y_pred in predictions:
        results['names'].append(name)
        results['explained_variance'].append(metrics.explained_variance_score(y_test, y_pred))
        results['mean_squared_log_error'].append(metrics.mean_squared_log_error(y_test, y_pred))
        results['r2'].append(metrics.r2_score(y_test, y_pred))
        results['mae'].append(metrics.mean_absolute_error(y_test, y_pred))
        results['mse'].append(metrics.mean_squared_error(y_test, y_pred))
        results['rmse'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    def plotBar(ax, x, y, title):
        ax.bar(x, y)
        ax.set_ylim(min(y) - (max(y)-min(y)) * 0.1, max(y) + (max(y)-min(y)) * 0.1)
        ax.set_xlabel(title)
        return ax

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8), sharex='row')
    fig.suptitle('Result Analysis')
    plotBar(ax1, results['names'], results['explained_variance'], 'Explained Variance')
    plotBar(ax2, results['names'], results['mean_squared_log_error'], 'Mean Squared Log Error')
    plotBar(ax3, results['names'], results['r2'], 'R2')
    plotBar(ax4, results['names'], results['mae'], 'MAE')
    plotBar(ax5, results['names'], results['mse'], 'MSE')
    plotBar(ax6, results['names'], results['rmse'], 'RMSE')
    plt.show()


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
    # plots a line graph of BTC True and Predicted Close
    plt.figure()
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(predictions) + 2)))
    plt.plot(X_train.index, y_train, c=next(colour), label='Train')
    plt.plot(X_test.index, y_test, c=next(colour), label='Test')
    for name, y_pred in predictions:
        plt.plot(X_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.title('BTC Predicted Close')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()

    # plots a closeup view if the test data and predictions
    plt.figure()
    colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(predictions) + 2)))
    plt.plot(X_test.index, y_test, c=next(colour), label='Test')
    for name, y_pred in predictions:
        plt.plot(X_test.index, y_pred, c=next(colour), label=f"{name} Predictions")
    plt.title('BTC Predicted Close (Closeup)')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.legend()
    plt.show()


def compareEstimators(dataset: Dataset, random_state: int = None) -> None:
    """
    Cross-validates each estimator model then plots the fitting times
    and test scores. Removes the poorly performing estimator then displays
    a result analysis and plots predictions.

    :param dataset:
    :param random_state:
    :return: None
    """
    # TODO: Fix documentation
    dataset = examples.processDataset(dataset)

    estimators = {'GBR': ensemble.GradientBoostingRegressor(random_state=random_state),
                  'RFR': ensemble.RandomForestRegressor(random_state=random_state),
                  'KNR': neighbors.KNeighborsRegressor(),
                  'MLPR': neural_network.MLPRegressor(random_state=random_state),
                  'SVR': svm.SVR(),
                  'DTR': tree.DecisionTreeRegressor(random_state=random_state),
                  'ETR': tree.ExtraTreeRegressor(random_state=random_state)}

    X_train, X_test, y_train, y_test = dataset.split(random_state=random_state, shuffle=False)

    results = compareModels(estimators, X_train, y_train)

    plt.figure()
    scores = [results[name]['test_score'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Test Score Comparison')

    plt.figure()
    scores = [results[name]['fit_time'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Fit Time Comparison')
    plt.show()

    # removes estimators that performed poorly
    del results['KNR']
    del results['MLPR']
    del results['SVR']

    plt.figure()
    scores = [results[name]['test_score'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Test Score Comparison')

    plt.figure()
    scores = [results[name]['fit_time'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Fit Time Comparison')
    plt.show()

    predictions = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        y_pred = results[name]['model'].predict(X_test)
        predictions.append((name, y_pred))

    plotEstimatorResultAnalysis(y_test, predictions)

    plotPredictions(X_train, X_test, y_train, y_test, predictions)


def plotClassifierResultAnalysis(y_test, predictions):
    results = {'names': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for name, y_pred in predictions:
        results['names'].append(name)
        results['accuracy'].append(metrics.accuracy_score(y_test, y_pred))
        results['precision'].append(metrics.precision_score(y_test, y_pred))
        results['recall'].append(metrics.recall_score(y_test, y_pred))
        results['f1'].append(metrics.f1_score(y_test, y_pred))

    def plotBar(ax, x, y, title):
        ax.bar(x, y)
        ax.set_ylim(min(y) - (max(y)-min(y)) * 0.1, max(y) + (max(y)-min(y)) * 0.1)
        ax.set_xlabel(title)
        return ax

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex='row')
    fig.suptitle('Result Analysis')
    plotBar(ax1, results['names'], results['accuracy'], 'Accuracy')
    plotBar(ax2, results['names'], results['precision'], 'Precision')
    plotBar(ax3, results['names'], results['recall'], 'Recall')
    plotBar(ax4, results['names'], results['f1'], 'F1')
    plt.show()


def plotClassifications(y_test: DataFrame, names: list, predictions: list) -> None:
    if len(predictions) != 4:
        logging.warning(f"Incorrect number of names and predictions")
        return

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
    dataset.apply(examples.convertToCategorical)

    X_train, X_test, y_train, y_test = dataset.split(train_size=0.1, random_state=random_state, shuffle=False)


    models = {'GBC': ensemble.GradientBoostingClassifier(random_state=random_state),
              'RFC': ensemble.RandomForestClassifier(random_state=random_state),
              'LR': linear_model.LogisticRegression(random_state=random_state),
              'RC': linear_model.RidgeClassifier(random_state=random_state),
              'SGDC': linear_model.SGDClassifier(random_state=random_state),
              'KNC': neighbors.KNeighborsClassifier(),
              'NC': neighbors.NearestCentroid(),
              'MLPC': neural_network.MLPClassifier(random_state=random_state),
              'SVC': svm.SVC(),
              'DTC': tree.DecisionTreeClassifier(random_state=random_state)}

    results = compareModels(models, X_train, y_train)

    plt.figure()
    scores = [results[name]['test_score'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Test Score Comparison')

    plt.figure()
    scores = [results[name]['fit_time'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Fit Time Comparison')
    plt.show()

    del results['LR']
    del results['SGDC']
    del results['KNC']
    del results['NC']
    del results['MLPC']
    del results['SVC']

    plt.figure()
    scores = [results[name]['test_score'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Test Score Comparison')

    plt.figure()
    scores = [results[name]['fit_time'] for name in results]
    plt.boxplot(scores, labels=[name for name in results])
    plt.title('Algorithm Fit Time Comparison')
    plt.show()

    predictions = []
    for name in results:
        results[name]['model'].fit(X_train, y_train)
        predictions.append((name, results[name]['model'].predict(X_test)))

    plotClassifierResultAnalysis(y_test, predictions)

    plotClassifications(y_test, *zip(*predictions))
