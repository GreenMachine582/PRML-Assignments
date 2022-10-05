from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import examples
import machine_learning as ml


# Constants
local_dir = os.path.dirname(__file__)


def processData(df: DataFrame, target) -> DataFrame:
    """

    :param df: DataFrame
    :param target:
    :return:
        - dataset - DataFrame
    """
    # TODO: Fix docstring
    logging.info("Processing data")

    df[target] = df[target].astype('category')

    x = df.drop(target, axis=1)  # denotes independent features

    print(df.axes)
    print(df.head())
    print(df.isnull().sum())  # check for missing values
    print(df.dtypes)
    print(np.array(x.iloc[9].values).reshape(28, 28))
    print("X shape:", x.shape)
    print(df.corr())  # corresponding matrix
    return df


def exploratoryDataAnalysis(df: DataFrame, target) -> None:
    """

    :param df: DataFrame
    :param target:
    :return:
        - None
    """
    # TODO: Fix docstring
    #   Add legend to scatter plot
    logging.info("Exploratory Data Analysis")

    y = df[target]

    # plots a corresponding matrix
    # plt.figure()
    # sn.heatmap(dataset.corr(), annot=True)

    # plots an example for each target
    plt.figure(figsize=(15, 3))
    for i in range(10):
        for idx in range(df.shape[0]):
            instance_data = df.iloc[idx]
            label = instance_data[-1]
            if label == i:
                image = instance_data[:-1].values
                plt.subplot(1, 10, i + 1)
                plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray)
                plt.title("Label: %i\n" % int(i), fontsize=15)
                break
    plt.plot()

    # plots a bar graph to represent number of instances per target
    plt.figure()
    df[target].value_counts().plot(kind="bar")

    # plots a PCA scatter plot to represent all instances and targets
    plt.figure(figsize=(15, 5))
    for i in np.unique(y):
        n = df[y == i]
        pca = PCA(n_components=2)
        view = pca.fit_transform(n)
        plt.scatter(view[:, 0], view[:, 1], label=i, alpha=0.2, cmap="Set1")
    plt.legend()
    plt.plot()

    plt.show()  # displays the plots


def resultAnalysis(model: Any, X: DataFrame, X_test: DataFrame, y_test: DataFrame) -> None:
    """

    :param model: Any
    :param X: DataFrame
    :param X_test: DataFrame
    :param y_test: DataFrame
    :return:
        - None
    """
    # TODO: Fix docstring
    logging.info("Analysing results")
    score = model.score(X_test, y_test)
    print("Score - %.4f%s" % (score * 100, "%"))

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    # plots the first 5 predictions
    plt.figure(label="Predictions", figsize=(10, 3))
    for i in range(5):
        prediction = y_pred[i]
        plt.subplot(1, 5, i + 1)
        plt.axis("off")
        image = X.iloc[i].values
        plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title("Prediction: %i" % int(prediction))
    plt.plot()

    # finds the misclassified and correctly classified predictions
    misclassified_indexes, classified_indexes = [], []
    for i in range(y_test.shape[0]):
        label, predict = y_test.iloc[i], y_pred[i]
        classified_indexes.append(i) if label == predict else misclassified_indexes.append(i)

    # plots the misclassified predictions
    plt.figure(label="Misclassified predictions", figsize=(15, 3))
    for plot_index, bad_index in enumerate(misclassified_indexes[:5]):
        plt.subplot(1, 5, plot_index + 1)
        plt.axis("off")
        image = X_test.iloc[bad_index].values
        plt.imshow(np.array(image).reshape(28, 28),
                   cmap=plt.cm.gray, interpolation="nearest")
        plt.title("Predicted: {}, Actual: {}".format(y_pred[bad_index], np.array(y_test)[bad_index]),
                  fontsize=15)
    plt.plot()

    # plots the correctly classified predictions
    plt.figure(label="Correctly classified predictions", figsize=(15, 3))
    for plot_index, good_index in enumerate(classified_indexes[:5]):
        plt.subplot(1, 5, plot_index + 1)
        plt.axis("off")
        image = X_test.iloc[good_index].values
        plt.imshow(np.array(image).reshape(28, 28),
                   cmap=plt.cm.gray, interpolation="nearest")
        plt.title("Predicted: {}, Actual: {}".format(y_pred[good_index], np.array(y_test)[good_index]),
                  fontsize=15)
    plt.plot()

    plt.show()  # displays all plots

    print(classification_report(y_test, y_pred))

    examples.estimator.resultAnalysis(y_test, y_pred)


def main(dir_=local_dir):
    config = ml.Config(dir_, 'Fashion-MNIST')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset.apply(processData, dataset.target)

    exploratoryDataAnalysis(dataset.df, dataset.target)

    X, y = dataset.getIndependent(), dataset.getDependent()
    X_train, X_test, y_train, y_test = dataset.split(random_state=config.random_state, shuffle=False)

    model = ml.Model(config.model, model=LogisticRegression(solver="lbfgs", max_iter=100))

    model.model.fit(X_train, y_train)
    model.save()

    resultAnalysis(model.model, X, X_test, y_test)
    return


if __name__ == '__main__':
    main()
