from __future__ import annotations

import logging
import os

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DataFrame: type = pd.core.frame.DataFrame


class MachineLearning(object):

    def __init__(self, config: Config, dataset_name: str, names: list = None):
        self.config = config
        self.dataset_name = dataset_name
        self.names = names

        self.raw_dataset = None

        self.X = None
        self.y = None
        self.dataset = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None

    def loadDataset(self, dataset_name: str) -> Any:
        """
        Loads a dataset from a CSV file or fetches from OpenML and converts the dataset
        to a Pandas DataFrame.
        :param dataset_name: str
        :return:
            - dataset - Any
        """
        logging.info(f"Loading dataset")
        if '.csv' in dataset_name:
            if not os.path.isfile(dataset_name):
                logging.error(f"Missing file")
                return
            dataset = pd.read_csv(dataset_name, names=self.config.names, sep=self.config.seperator)
            logging.info('Loaded dataset')
        else:
            dataset = fetch_openml(dataset_name, version=1)
            logging.info('Fetched dataset')
        return dataset

    def saveDataset(self, dataset_name: str, dataset: Any) -> None:
        """
        Saves the dataset in a csv file using a Pandas method.
        :param dataset_name: str
        :param dataset: Any
        :return:
            - None
        """
        dataset.to_csv(dataset_name, sep=self.config.seperator, index=False)
        logging.info(f"Dataset saves {dataset_name}")

    def processData(self, dataset: Any) -> tuple:
        if isinstance(dataset, DataFrame):
            X, y = dataset.drop(self.config.target, axis=1), dataset[self.config.target]
            print(dataset.axes)
            print(dataset.head())
            print(dataset.dtypes)
        else:
            X, y = dataset.data, dataset.target
            print(dataset.keys())
            print(dataset.target[:5])
            try:
                print(np.array(X.iloc[9, :]).reshape(28, 28))
            except AttributeError:
                print(np.array(X[9, :]).reshape(28, 28))
            print(X.shape)

        logging.info('Data processed')
        return dataset, X, y

    def extractFeatures(self, dataset: Any) -> tuple:
        if isinstance(dataset, DataFrame):
            X, y = dataset.drop(self.config.target, axis=1), dataset[self.config.target]
        else:
            X, y = dataset.data, dataset.target

        logging.info('Features extracted')
        return dataset, X, y

    def splitDataset(self, x, y) -> tuple:
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=self.config.split_ratio,
                                                            random_state=self.config.random_seed)
        logging.info('Dataset split')
        return X_train, X_test, y_train, y_test

    def trainModel(self, x_train, y_train):
        model = None
        if self.config.model_type == 'LogisticRegression':
            model = LogisticRegression(solver='lbfgs', max_iter=100)
            model.fit(x_train, y_train)
        return model

    @staticmethod
    def resultAnalysis(model, x, x_test, y_test):
        score = model.score(x_test, y_test)
        print("Score - %.2f%s" % (score * 100, '%'))

        y_pred = model.predict(x_test)

        print(confusion_matrix(y_test, y_pred))

        plot_confusion_matrix(model, x_test, y_test)
        plt.show()

        plt.figure(figsize=(10, 2))
        for i in range(5):
            prediction = model.predict(x)[i]
            plt.subplot(1, 5, i + 1)
            plt.axis("off")
            try:
                image = x.iloc[i, :]
            except AttributeError:
                image = x[i, :]
            plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % int(prediction))
        plt.show()

        index = 0
        misclassifiedIndexes = []
        for label, predict in zip(y_test, y_pred):
            if label != predict:
                misclassifiedIndexes.append(index)
            index += 1

        plt.figure(figsize=(20, 3))
        for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
            plt.subplot(1, 5, plotIndex + 1)
            plt.axis("off")
            try:
                image = x_test.iloc[badIndex, :]
            except AttributeError:
                image = x_test[badIndex, :]
            plt.imshow(np.array(image).reshape(28, 28),
                       cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex],
                                                         np.array(y_test)[badIndex]),
                      fontsize=20)

    # def train(self):
    #     logging.info('Training models')
    #     self.models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    #                    ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier())]
    #     # evaluate each model in turn
    #     results = []
    #     names = []
    #     for name, model in self.models:
    #         kfold = KFold(n_splits=10, random_state=self.config.random_seed, shuffle=True)
    #         cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
    #         results.append(cv_results)
    #         names.append(name)
    #         msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #         print(msg)
    #         logging.info(f"{name} model trained")
    #     logging.info('Training complete')
    #     return results, names
    #
    # def results(self, results, names):
    #     # Compare Algorithms
    #     fig = pyplot.figure()
    #     fig.suptitle('Algorithm Comparison')
    #     ax = fig.add_subplot(111)
    #     pyplot.boxplot(results)
    #     ax.set_xticklabels(names)
    #     pyplot.show()
    #
    #     # Make predictions on validation dataset
    #     knn = KNeighborsClassifier()
    #     knn.fit(self.X_train, self.y_train)
    #     predictions = knn.predict(self.X_test)
    #     print(accuracy_score(self.y_test, predictions))
    #     print(confusion_matrix(self.y_test, predictions))
    #     print(classification_report(self.y_test, predictions))
    #     logging.info('Results complete')

    @staticmethod
    def saveModel(model_dir, model):
        pickle.dump(model, open(model_dir, 'wb'))

    @staticmethod
    def loadModel(model_dir):
        loaded_model = pickle.load(open(model_dir, 'rb'))
        return loaded_model

    def main(self):
        self.raw_dataset = self.loadDataset(self.dataset_name)
        if self.raw_dataset is None:
            logging.error("Couldn't load a dataset")
            return

        self.dataset, self.X, self.y = self.processData(self.raw_dataset)
        self.dataset, self.X, self.y = self.extractFeatures(self.dataset)

        self.X_train, self.X_test, self.y_train, self.y_test = self.splitDataset(self.X, self.y)

        self.model = self.trainModel(self.X_train, self.y_train)
        self.resultAnalysis(self.model, self.X, self.X_test, self.y_test)

        # results, names = self.train()
        # self.results(results, names)

        logging.info('Competed')
