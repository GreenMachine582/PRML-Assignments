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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay,\
    mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DataFrame: type = pd.core.frame.DataFrame


class MachineLearning(object):

    def __init__(self, config: Config, dataset_name: str, names: list = None):
        self.config = config
        self.dataset_name = dataset_name
        self.names = names

        self.raw_dataset = self.loadDataset(dataset_name)

        self.X = None
        self.y = None
        self.dataset = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None

    def bunchToDataframe(self, fetched_dataset: Bunch) -> DataFrame:
        """
        Creates a pandas DataFrame dataset from SKLearn Bunch object.
        :param fetched_dataset: Bunch
        :return:
            - dataset - DataFrame
        """
        dataset = pd.DataFrame(data=fetched_dataset['data'], columns=fetched_dataset['feature_names'])
        dataset[self.config.target] = fetched_dataset['target']
        return dataset

    def loadDataset(self, dataset_name: str) -> DataFrame:
        """
        Checks and loads a locally stored .csv dataset as a pandas DataFrame.
        If dataset was not located, it will attempt to fetch from OpenML, and convert
        the dataset to a DataFrame object.
        :param dataset_name: str
        :return:
            - dataset - DataFrame
        """
        logging.info(f"Loading dataset")

        if '.csv' not in dataset_name:
            if not os.path.isfile(f'{self.config.dataset_dir}{dataset_name}.csv'):
                fetched_dataset = fetch_openml(dataset_name, version=1)
                logging.info('Fetched dataset')
                try:
                    dataset = self.bunchToDataframe(fetched_dataset)
                    self.config.names = fetched_dataset['feature_names'] + [self.config.target]
                    logging.info('Converted dataset to DataFrame')
                    dataset_name = dataset_name + '.csv'
                    self.saveDataset(dataset_name, dataset)
                except Exception as e:
                    logging.warning("Couldn't fetch dataset", e)
                    return

        if not os.path.isfile(self.config.dataset_dir + dataset_name):
            logging.error(f"Missing file {self.config.dataset_dir}{dataset_name}")
            return
        dataset = pd.read_csv(self.config.dataset_dir + dataset_name, names=self.config.names,
                              sep=self.config.seperator)
        logging.info('Loaded dataset')
        return dataset

    def saveDataset(self, dataset_name: str, dataset: DataFrame) -> None:
        """
        Saves the dataset in a .csv file using a Pandas method.
        :param dataset_name: str
        :param dataset: DataFrame
        :return:
            - None
        """
        dataset.to_csv(self.config.dataset_dir + dataset_name, sep=self.config.seperator, index=False)
        logging.info(f"Dataset saved {self.config.dataset_dir + dataset_name}")

    def processData(self, dataset: DataFrame) -> tuple:
        """

        :param dataset: DataFrame
        :return:
            - dataset, X, y - tuple[DataFrame]
        """
        plt.figure(figsize=(10, 3))
        for i in range(5):
            instance_data = dataset.iloc[i]
            image = instance_data[:-1].values
            label = instance_data[-1]
            plt.subplot(1, 5, i + 1)
            plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray)
            plt.title('Training: %i\n' % int(label), fontsize=15)
        plt.plot()
        logging.info('Example instances')

        plt.figure()
        dataset[self.config.target].value_counts().plot(kind='bar')

        plt.figure(figsize=(15, 3))
        for i in range(10):
            for idx in range(dataset.shape[0]):
                instance_data = dataset.iloc[idx]
                label = instance_data[-1]
                if label == i:
                    image = instance_data[:-1].values
                    plt.subplot(1, 10, i + 1)
                    plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray)
                    plt.title('Label: %i\n' % int(i), fontsize=15)
                    break
        plt.plot()

        plt.show()

        X, y = dataset.drop(self.config.target, axis=1), dataset[self.config.target]

        if self.config.show_small_responses:
            print(dataset.axes)
            print(dataset.head())
            print(dataset.dtypes)

            print(np.array(X.iloc[9][:-1].values.reshape(28, 28)))
        print('X shape:', X.shape)
        quit()
        logging.info('Data processed')
        return dataset, X, y

    def extractFeatures(self, dataset: DataFrame) -> tuple:
        X, y = dataset.drop(self.config.target, axis=1), dataset[self.config.target]

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
        print("Score - %.4f%s" % (score * 100, '%'))

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

        plt.figure(figsize=(10, 3))
        for i in range(5):
            prediction = model.predict(x)[i]
            plt.subplot(1, 5, i + 1)
            plt.axis("off")
            image = x.iloc[[i]].values[0]
            plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title('Prediction: %i' % int(prediction))
        plt.plot()
        logging.info('Predicted results')

        index = 0
        misclassifiedIndexes = []
        for label, predict in zip(y_test, y_pred):
            if label != predict:
                misclassifiedIndexes.append(index)
            index += 1

        plt.figure(figsize=(10, 3))
        for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
            plt.subplot(1, 5, plotIndex + 1)
            plt.axis("off")
            image = x_test.iloc[[badIndex]].values[0]
            plt.imshow(np.array(image).reshape(28, 28),
                       cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex], np.array(y_test)[badIndex]),
                      fontsize=15)
        plt.plot()
        logging.info('Misclassified results')

        print(classification_report(y_test, y_pred))

        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
        plt.show()
        logging.info('Results complete')

    def saveModel(self, model_name: str, model: Any) -> None:
        """
        Saves the model by serialising the model object.
        :param model_name: str
        :param model: Any
        :return:
            - None
        """
        try:
            pickle.dump(model, open(self.config.model_dir + model_name, 'wb'))
        except Exception as e:
            logging.warning(e)

    def loadModel(self, model_name: str) -> Any | None:
        """
        Loads the model by deserialising a json file
        :param model_name: str
        :return:
            - model - Any | None
        """
        try:
            return pickle.load(open(self.config.model_dir + model_name, 'rb'))
        except Exception as e:
            logging.warning(e)

    def main(self):
        if self.raw_dataset is None:
            logging.error("Couldn't load a dataset")
            return

        self.dataset, self.X, self.y = self.processData(self.raw_dataset)
        self.dataset, self.X, self.y = self.extractFeatures(self.dataset)

        self.X_train, self.X_test, self.y_train, self.y_test = self.splitDataset(self.X, self.y)

        self.model = self.trainModel(self.X_train, self.y_train)
        self.resultAnalysis(self.model, self.X, self.X_test, self.y_test)

        logging.info('Saving model')
        self.saveModel('iris.model', self.model)
        logging.info('Loading model')
        self.model = self.loadModel('iris.model')
        self.resultAnalysis(self.model, self.X, self.X_test, self.y_test)

        logging.info('Competed')
