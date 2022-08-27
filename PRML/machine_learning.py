from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


def loadDataset(dataset_name: str, seperator: str = ';') -> Any:
    """
    Loads a dataset from a csv file or fetches from openml.
    :param dataset_name: str
    :param seperator: str
    :return:
        - dataset - Any
    """
    logging.info('Loading dataset')
    if '.csv' in dataset_name:
        if not os.path.isfile(dataset_name):
            logging.error(f"Missing file {dataset_name}")
            return
        dataset = pd.read_csv(dataset_name, sep=seperator)
        logging.info('Loaded dataset')
    else:
        dataset = fetch_openml(dataset_name, version=1)
        logging.info('Fetched dataset')
    return dataset


def saveDataset(dataset_name: str, dataset: Any) -> None:
    # Save the dataset into a .csv file for easy access in future and reviewing.
    pass


def processData(dataset):
    # print(dataset.keys())
    # print(dataset.target[:5])

    plt.figure(figsize=(10, 2))
    for i in range(5):
        _image = dataset.data[i, :]
        _label = dataset.target[i]
        plt.subplot(1, 5, i + 1)
        plt.imshow(np.array(_image).reshape(28, 28), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % int(_label), fontsize=15)

    # print(dataset.data)
    # print(np.array(dataset.data[9, :]).reshape(28, 28))

    # X, y = dataset.data, dataset.target

    # print(y[0])
    # n_samples = len(X)
    # print('Instances:', n_samples)
    # print('Data Size:', X.shape)

    return dataset


def extractFeatures(dataset):
    X, y = dataset.data, dataset.target
    return dataset


def splitDataset(dataset):
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test


def trainModel(model_type, datasets):
    X_train, y_train = datasets[0], datasets[2]
    model = None
    if model_type == 'LogisticRegression':
        model = LogisticRegression(solver='lbfgs', max_iter=100)
        model.fit(X_train, y_train)
    return model


def testModel(model, datasets):
    X_test = datasets[1]

    y_pred = model.predict(X_test)
    print(y_pred)
    return y_pred


def resultAnalysis(model, datasets, dataset):
    X_test, y_test = datasets[1], datasets[3]
    score = model.score(X_test, y_test)
    print("Score - %.2f%s" % (score * 100, '%'))

    y_pred = testModel(model, datasets)

    print(confusion_matrix(y_test, y_pred))

    X, y = dataset.data, dataset.target

    images_and_predictions = list(zip(dataset.data, model.predict(X)))
    print(images_and_predictions)

    plt.figure(figsize=(10, 2))
    for i in range(5):
        image = dataset.data[i, :]
        prediction = model.predict(X)[i]
        plt.subplot(1, 5, i + 1)
        plt.axis("off")
        plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.title('Prediction: %i' % int(prediction))
    plt.show()

    index = 0
    misclassifiedIndexes = []
    for label, predict in zip(y_test, y_pred):
        if label != predict:
            misclassifiedIndexes.append(index)
        index += 1

    # print(misclassifiedIndexes[:5])
    #
    # print(y_pred[4])
    #
    # print(np.array(y_test)[:5])

    plt.figure(figsize=(20, 3))
    for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
        plt.subplot(1, 5, plotIndex + 1)
        plt.axis("off")
        plt.imshow(np.array(X_test[badIndex, :]).reshape(28, 28),
                   cmap=plt.cm.gray, interpolation='nearest')
        plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex],
                                                     np.array(y_test)[badIndex]),
                  fontsize=20)


def main(dataset_name, model_type: str = 'LogisticRegression'):
    raw_dataset = loadDataset(dataset_name)
    if raw_dataset is None:
        return
    processed_dataset = processData(raw_dataset)
    dataset = extractFeatures(processed_dataset)

    saveDataset(dataset_name, dataset)

    datasets = splitDataset(dataset)

    model = trainModel(model_type, datasets)

    resultAnalysis(model, datasets, dataset)
