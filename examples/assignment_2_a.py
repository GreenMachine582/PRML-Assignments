
import logging
import os

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor

import examples
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def plotPredictions(df: DataFrame, X_test: DataFrame, y_pred: DataFrame) -> None:
    """
    Plots the BTC daily demand and predictions.

    :param df: the dataset itself, should be a DataFrame
    :param X_test: testing independent features, should be a DataFrame
    :param y_pred: predicted dependent variables, should be a DataFrame
    :return:
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

    X, y = dataset.getIndependent(), dataset.getDependent()
    X_train, X_test, y_train, y_test = dataset.split(config.random_seed)

    estimator = GradientBoostingRegressor(random_state=config.random_seed)
    model = ml.Model(config.model, estimator=estimator)

    param_grid = {'loss': ['squared_error', 'absolute_error']}

    cv_results = model.gridSearch(param_grid, X, y)
    estimator = cv_results.best_estimator_
    print('The best estimator:', estimator)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    model.fit(X_train, y_train)
    model.save()

    y_pred = model.predict(X_test)
    ml.resultAnalysis(y_test, y_pred)
    plotPredictions(dataset.df, X_test, y_pred)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
