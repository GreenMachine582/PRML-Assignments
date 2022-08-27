from __future__ import annotations

import logging
import os

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DataFrame: type = pd.core.frame.DataFrame


def loadDataset(dataset_name: str, config: Config) -> Any:
    """
    Loads a dataset from a CSV file or fetches from OpenML and converts the dataset
    to a Pandas DataFrame.
    :param dataset_name: str
    :param config: Config
    :return:
        - dataset - Any
    """
    logging.info(f"Loading dataset")
    if '.csv' in dataset_name:
        if not os.path.isfile(f"{config.dataset_dir}{dataset_name}"):
            logging.error(f"Missing file")
            return
        dataset = pd.read_csv(f"{config.dataset_dir}{dataset_name}", names=config.names, sep=config.seperator)
        logging.info('Loaded dataset')
    else:
        dataset = fetch_openml(dataset_name, version=1)
        logging.info('Fetched dataset')
    return dataset


def saveDataset(dataset_dir: str, dataset_name: str, dataset: Any) -> None:
    """
    Saves the dataset in a csv file using a Pandas method.
    :param dataset_dir: str
    :param dataset_name: str
    :param dataset: Any
    :return:
        - None
    """
    dataset.to_csv(f"{dataset_dir}{dataset_name}", sep='\t', index=False)
    logging.info(f"Dataset saves {dataset_dir}{dataset_name}")


class MachineLearning(object):

    def __init__(self, config: Config, dataset_name: str, names: list = None):
        self.config = config
        self.dataset_name = dataset_name
        self.names = names

        self.raw_dataset = loadDataset(dataset_name, self.config)
        if self.raw_dataset is None:
            logging.error("Couldn't load a dataset")

        self.X = None
        self.y = None
        self.dataset = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = None

    def processData(self):
        self.dataset = self.raw_dataset
        if isinstance(self.dataset, DataFrame):
            self.X, self.y = self.dataset.drop(self.config.target, axis=1), self.dataset[self.config.target]
            print(self.dataset.axes)
            print(self.dataset.head())
            print(self.dataset.dtypes)
        else:
            self.X, self.y = self.dataset.data, self.dataset.target
            print(self.dataset.keys())
            print(self.dataset.target[:5])
            try:
                print(np.array(self.X.iloc[9, :]).reshape(28, 28))
            except AttributeError:
                print(np.array(self.X[9, :]).reshape(28, 28))
            print(self.X.shape)

        logging.info('Data processed')

    def extractFeatures(self) -> None:
        if isinstance(self.dataset, DataFrame):
            pass
        else:
            pass
        logging.info('Features extracted')

    def splitDataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                train_size=self.config.split_ratio,
                                                                                random_state=self.config.random_seed)
        logging.info('Dataset split')

    def trainModel(self):
        if self.config.model_type == 'LogisticRegression':
            self.models = LogisticRegression(solver='lbfgs', max_iter=100)
            self.models.fit(self.X_train, self.y_train)

    def resultAnalysis(self):
        score = self.models.score(self.X_test, self.y_test)
        print("Score - %.2f%s" % (score * 100, '%'))

        y_pred = self.models.predict(self.X_test)

        print(confusion_matrix(self.y_test, y_pred))

        # Doesn't work with Fashion-MNIST
        plt.figure(figsize=(10, 2))
        for i in range(5):
            prediction = self.models.predict(self.X)[i]
            plt.subplot(1, 5, i + 1)
            plt.axis("off")
            try:
                image = self.X.iloc[i, :]
            except AttributeError:
                image = self.X[i, :]
            plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % int(prediction))
        plt.show()

        index = 0
        misclassifiedIndexes = []
        for label, predict in zip(self.y_test, y_pred):
            if label != predict:
                misclassifiedIndexes.append(index)
            index += 1

        plt.figure(figsize=(20, 3))
        for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
            plt.subplot(1, 5, plotIndex + 1)
            plt.axis("off")
            try:
                image = self.X_test.iloc[badIndex, :]
            except AttributeError:
                image = self.X_test[badIndex, :]
            plt.imshow(np.array(image).reshape(28, 28),
                       cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex],
                                                         np.array(self.y_test)[badIndex]),
                      fontsize=20)

    def train(self):
        logging.info('Training models')
        self.models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
                       ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier())]
        # evaluate each model in turn
        results = []
        names = []
        for name, model in self.models:
            kfold = KFold(n_splits=10, random_state=self.config.random_seed, shuffle=True)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
            logging.info(f"{name} model trained")
        logging.info('Training complete')
        return results, names

    def results(self, results, names):
        # Compare Algorithms
        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()

        # Make predictions on validation dataset
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        print(accuracy_score(self.y_test, predictions))
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))
        logging.info('Results complete')

    def main(self):
        if self.raw_dataset is None:
            logging.error("Couldn't load a dataset")
            return

        self.processData()
        # self.extractFeatures()

        self.splitDataset()

        # self.trainModel()
        # self.resultAnalysis()

        results, names = self.train()
        self.results(results, names)

        logging.info('Competed')
