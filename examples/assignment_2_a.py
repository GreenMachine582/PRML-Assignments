
import logging
import os

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from numpy import ndarray
from pandas import DataFrame
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor

import examples
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def plotPredictions(df: DataFrame, X_test: DataFrame, y_pred: ndarray) -> None:
    """
    Plots the BTC daily demand and predictions.

    :param df: the dataset itself, should be a DataFrame
    :param X_test: testing independent features, should be a DataFrame
    :param y_pred: predicted dependent variables, should be a ndarray
    :return: None
    """
    # plots a line graph of BTC True and Predicted Close vs Date
    plt.figure()
    plt.plot(df.index, df['Close'], color='blue')
    plt.plot(X_test.index, y_pred, color='red')

    plt.title('BTC Close Vs Date')
    plt.xlabel('Date')
    plt.ylabel('Close ($USD)')
    plt.show()


def main(dir_=local_dir):
    config = ml.Config(dir_, 'BTC-USD')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = examples.processDataset(dataset)

    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    model = ml.Model(config.model, model=GradientBoostingRegressor(random_state=config.random_state))
    param_grid = {'loss': ['squared_error', 'absolute_error']}

    cv_results = model.gridSearch(param_grid, X_train, y_train)
    print('The best estimator:', cv_results.best_estimator_)
    print('The best score:', cv_results.best_score_)
    print('The best params:', cv_results.best_params_)

    model.update(model=GradientBoostingRegressor(random_state=config.random_state, **cv_results.best_params_))
    model.fit(X_train, y_train)

    model.save()

    y_pred = model.predict(X_test)
    ml.resultAnalysis(y_test, y_pred)
    plotPredictions(dataset.df, X_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
