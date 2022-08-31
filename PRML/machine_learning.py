from __future__ import annotations

import logging
import os

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay,\
    mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


class MachineLearning(object):

    def __init__(self, config: Config):
        self.config = config

        self.raw_dataset = self.loadDataset()

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
        Creates a pandas DataFrame dataset from the SKLearn Bunch object.
        :param fetched_dataset: Bunch
        :return:
            - dataset - DataFrame
        """
        logging.info('Converting Bunch to DataFrame')
        dataset = pd.DataFrame(data=fetched_dataset['data'], columns=fetched_dataset['feature_names'])
        dataset[self.config.target] = fetched_dataset['target']
        return dataset

    def loadDataset(self) -> DataFrame:
        """
        Checks and loads a locally stored .csv dataset as a pandas DataFrame.
        If dataset was not located, it will attempt to fetch from OpenML, and convert
        the dataset to a DataFrame object.
        :return:
            - dataset - DataFrame
        """

        if not os.path.isfile(self.config.dataset_dir):
            logging.info('Fetching dataset')
            try:
                fetched_dataset = fetch_openml(self.config.dataset_name, version=1)
            except Exception as e:
                logging.warning(e)
                return
            dataset = self.bunchToDataframe(fetched_dataset)
            self.config.names = fetched_dataset['feature_names'] + [self.config.target]
            self.saveDataset(dataset)
        logging.info(f'Loading dataset {self.config.dataset_dir}')
        return pd.read_csv(self.config.dataset_dir, names=self.config.names, sep=self.config.seperator,
                           low_memory=False)

    def saveDataset(self, dataset: DataFrame) -> None:
        """
        Saves the dataset to a csv file using pandas.
        :param dataset: DataFrame
        :return:
            - None
        """
        logging.info(f'Saving dataset {self.config.dataset_dir}')
        dataset.to_csv(self.config.dataset_dir, sep=self.config.seperator, index=False)

    def processData(self, dataset: DataFrame) -> tuple:
        """

        :param dataset: DataFrame
        :return:
            - dataset, X, y - tuple[DataFrame]
        """
        # TODO: Fix docstring
        logging.info('Processing data')

        X = dataset.drop(self.config.target, axis=1)  # denotes independent features
        y = dataset[self.config.target]  # denotes dependent variables

        if self.config.show_small_responses:
            print(dataset.axes)
            print(dataset.head())
            print(dataset.isnull().sum())  # check for missing values
            print(dataset.dtypes)
            if self.config.dataset_name == 'Fashion-MNIST':
                print(np.array(X.iloc[9].values).reshape(28, 28))
            print('X shape:', X.shape)
            print(dataset.corr())  # corresponding matrix
        return dataset, X, y

    def extractFeatures(self, dataset: DataFrame, x: DataFrame, y: DataFrame) -> tuple:
        """

        :param dataset: DataFrame
        :param x: DataFrame
        :param y: DataFrame
        :return:
            - dataset, X, y - tuple[DataFrame]
        """
        # TODO: Fix docstring
        #   Add extraction methods
        logging.info('Extracting features')

        return dataset, x, y

    def exploratoryDataAnalysis(self, dataset: DataFrame, x: DataFrame, y: DataFrame) -> None:
        """

        :param dataset: DataFrame
        :param x: DataFrame
        :param y: DataFrame
        :return:
            - None
        """
        # TODO: Fix docstring
        #   Add legend to scatter plot
        logging.info('Exploratory Data Analysis')
        if not self.config.show_figs:
            return

        # plots a corresponding matrix
        # plt.figure()
        # sn.heatmap(dataset.corr(), annot=True)

        # plots an example for each target
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

        # plots a bar graph to represent number of instances per target
        plt.figure()
        dataset[self.config.target].value_counts().plot(kind='bar')

        # plots a PCA scatter plot to represent all instances and targets
        plt.figure(figsize=(15, 5))
        for i in np.unique(y):
            n = dataset[y == i]
            pca = PCA(n_components=2)
            view = pca.fit_transform(n)
            plt.scatter(view[:, 0], view[:, 1], label=i, alpha=0.2, cmap='Set1')
        plt.legend()
        plt.plot()

        plt.show()  # displays the plots

    def splitDataset(self, x: DataFrame, y: DataFrame) -> tuple:
        """
        Splits the dataset into train and test with a given ratio to avoid overfitting
        the model.
        :param x: DataFrame
        :param y: DataFrame
        :return:
            - X_train, X_test, y_train, y_test - tuple[DataFrame]
        """
        logging.info('Splitting data')
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=self.config.split_ratio,
                                                            random_state=self.config.random_seed)
        return X_train, X_test, y_train, y_test

    def trainModel(self, x_train: DataFrame, y_train: DataFrame) -> Any:
        """

        :param x_train: DataFrame
        :param y_train: DataFrame
        :return:
            - model - Any
        """
        # TODO: Fix docstring
        #   Add other ML techniques
        logging.info('Training model')
        model = None
        if self.config.model_type == 'LogisticRegression':
            model = LogisticRegression(solver='lbfgs', max_iter=100)

        return model.fit(x_train, y_train)

    def resultAnalysis(self, model: Any, x: DataFrame, x_test: DataFrame, y_test: DataFrame) -> None:
        """

        :param model: Any
        :param x: DataFrame
        :param x_test: DataFrame
        :param y_test: DataFrame
        :return:
            - None
        """
        # TODO: Fix docstring
        logging.info('Analysing results')
        score = model.score(x_test, y_test)
        print("Score - %.4f%s" % (score * 100, '%'))

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        if self.config.show_figs:
            ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

            # plots the first 5 predictions
            plt.figure(label='Predictions', figsize=(10, 3))
            for i in range(5):
                prediction = model.predict(x)[i]
                plt.subplot(1, 5, i + 1)
                plt.axis("off")
                image = x.iloc[i].values
                plt.imshow(np.array(image).reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
                plt.title('Prediction: %i' % int(prediction))
            plt.plot()

            # finds the misclassified and correctly classified predictions
            index = 0
            misclassified_indexes, classified_indexes = [], []
            for label, predict in zip(y_test, y_pred):
                if label != predict:
                    misclassified_indexes.append(index)
                elif label == predict:
                    classified_indexes.append(index)
                index += 1
            print(misclassified_indexes[:5])
            print(classified_indexes[:5])

            # plots the misclassified predictions
            plt.figure(label='Misclassified predictions', figsize=(15, 3))
            for plot_index, bad_index in enumerate(misclassified_indexes[:5]):
                plt.subplot(1, 5, plot_index + 1)
                plt.axis("off")
                image = x_test.iloc[bad_index].values
                plt.imshow(np.array(image).reshape(28, 28),
                           cmap=plt.cm.gray, interpolation='nearest')
                plt.title('Predicted: {}, Actual: {}'.format(y_pred[plot_index], np.array(y_test)[bad_index]),
                          fontsize=15)
            plt.plot()

            # plots the correctly classified predictions
            plt.figure(label='Correctly classified predictions', figsize=(15, 3))
            for plot_index, good_index in enumerate(classified_indexes[:5]):
                plt.subplot(1, 5, plot_index + 1)
                plt.axis("off")
                image = x_test.iloc[good_index].values
                plt.imshow(np.array(image).reshape(28, 28),
                           cmap=plt.cm.gray, interpolation='nearest')
                plt.title('Predicted: {}, Actual: {}'.format(y_pred[plot_index], np.array(y_test)[good_index]),
                          fontsize=15)
            plt.plot()

            plt.show()  # displays all plots

        print(classification_report(y_test, y_pred))

        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    def saveModel(self, model: Any) -> None:
        """
        Saves the model by serialising the model object.
        :param model: Any
        :return:
            - None
        """
        logging.info(f'Saving model {self.config.model_dir}')
        pickle.dump(model, open(self.config.model_dir, 'wb'))

    def loadModel(self) -> Any:
        """
        Loads the model by deserialising a model file.
        :return:
            - model - Any
        """
        logging.info(f'Loading model {self.config.model_dir}')
        try:
            return pickle.load(open(self.config.model_dir, 'rb'))
        except FileNotFoundError as e:
            logging.warning(e)

    def main(self) -> None:
        if self.raw_dataset is None:
            logging.error("Couldn't load a dataset")
            return

        self.dataset, self.X, self.y = self.processData(self.raw_dataset)

        self.dataset, self.X, self.y = self.extractFeatures(self.dataset, self.X, self.y)

        # self.exploratoryDataAnalysis(self.dataset, self.X, self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = self.splitDataset(self.X, self.y)

        self.model = self.trainModel(self.X_train, self.y_train)

        self.resultAnalysis(self.model, self.X, self.X_test, self.y_test)

        # self.saveModel(self.model)
        # self.model = self.loadModel()
        # self.resultAnalysis(self.model, self.X, self.X_test, self.y_test)
