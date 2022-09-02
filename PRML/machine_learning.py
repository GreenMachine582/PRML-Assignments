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

    def bunchToDataframe(self, fetched_dataset: Bunch) -> DataFrame:
        """
        Creates a pandas DataFrame dataset from the SKLearn Bunch object.
        :param fetched_dataset: Bunch
        :return:
            - dataset - DataFrame
        """
        logging.info("Converting Bunch to DataFrame")
        dataset = pd.DataFrame(data=fetched_dataset["data"], columns=fetched_dataset["feature_names"])
        dataset[self.config.target] = fetched_dataset["target"]
        return dataset

    def loadDataset(self) -> DataFrame:
        """
        Checks and loads a locally stored .csv dataset as a pandas DataFrame. If dataset
        was not located, it will attempt to fetch from OpenML, and convert the dataset to
        a DataFrame object.
        :return:
            - dataset - DataFrame
        """
        path, exist = self.config.checkPath(f"{self.config.dataset_dir}\\{self.config.dataset_name}", ".csv")
        if not exist:
            logging.warning(f"Missing file '{path}'")
            logging.info("Fetching dataset from openml")
            try:
                fetched_dataset = fetch_openml(self.config.dataset_name, version=1)
            except Exception as e:
                logging.warning(e)
                return
            dataset = self.bunchToDataframe(fetched_dataset)
            self.saveDataset(dataset)
        logging.info(f"Loading dataset '{path}'")
        dataset = pd.read_csv(path, names=self.config.names, sep=self.config.seperator, low_memory=False)
        return dataset

    def saveDataset(self, dataset: DataFrame) -> None:
        """
        Saves the dataset to a csv file using pandas.
        :param dataset: DataFrame
        :return:
            - None
        """
        _, exist = self.config.checkPath(self.config.dataset_dir)
        if not exist:
            os.makedirs(self.config.dataset_dir)
        path, _ = self.config.checkPath(f"{self.config.dataset_dir}\\{self.config.dataset_name}", ".csv")

        logging.info(f"Saving file '{path}'")
        dataset.to_csv(path, sep=self.config.seperator, index=False)

    def processData(self, dataset: DataFrame) -> tuple:
        """

        :param dataset: DataFrame
        :return:
            - dataset, X, y - tuple[DataFrame]
        """
        # TODO: Fix docstring
        logging.info("Processing data")

        dataset[self.config.target] = dataset[self.config.target].astype('category')

        X = dataset.drop(self.config.target, axis=1)  # denotes independent features
        y = dataset[self.config.target]  # denotes dependent variables

        if self.config.show_small_responses:
            print(dataset.axes)
            print(dataset.head())
            print(dataset.isnull().sum())  # check for missing values
            print(dataset.dtypes)
            if self.config.dataset_name == "Fashion-MNIST":
                print(np.array(X.iloc[9].values).reshape(28, 28))
            print("X shape:", X.shape)
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
        logging.info("Extracting features")

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
        logging.info("Exploratory Data Analysis")
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
                    plt.title("Label: %i\n" % int(i), fontsize=15)
                    break
        plt.plot()

        # plots a bar graph to represent number of instances per target
        plt.figure()
        dataset[self.config.target].value_counts().plot(kind="bar")

        # plots a PCA scatter plot to represent all instances and targets
        plt.figure(figsize=(15, 5))
        for i in np.unique(y):
            n = dataset[y == i]
            pca = PCA(n_components=2)
            view = pca.fit_transform(n)
            plt.scatter(view[:, 0], view[:, 1], label=i, alpha=0.2, cmap="Set1")
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
        logging.info("Splitting data")
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
        logging.info("Training model")
        model = None
        if self.config.model_type == "LogisticRegression":
            model = LogisticRegression(solver="lbfgs", max_iter=100)

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
        logging.info("Analysing results")
        score = model.score(x_test, y_test)
        print("Score - %.4f%s" % (score * 100, "%"))

        y_pred = model.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        if self.config.show_figs:
            ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

            # plots the first 5 predictions
            plt.figure(label="Predictions", figsize=(10, 3))
            for i in range(5):
                prediction = model.predict(x)[i]
                plt.subplot(1, 5, i + 1)
                plt.axis("off")
                image = x.iloc[i].values
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
                image = x_test.iloc[bad_index].values
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
                image = x_test.iloc[good_index].values
                plt.imshow(np.array(image).reshape(28, 28),
                           cmap=plt.cm.gray, interpolation="nearest")
                plt.title("Predicted: {}, Actual: {}".format(y_pred[good_index], np.array(y_test)[good_index]),
                          fontsize=15)
            plt.plot()

            plt.show()  # displays all plots

        print(classification_report(y_test, y_pred))

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("Mean Root Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

    def loadModel(self) -> Any:
        """
        Loads the model by deserialising a model file.
        :return:
            - model - Any
        """
        path, exist = self.config.checkPath(f"{self.config.model_dir}\\{self.config.dataset_name}", ".model")
        if exist:
            logging.info(f"Loading model '{path}'")
            return pickle.load(open(path, "rb"))
        else:
            logging.warning(f"Missing file '{path}'")

    def saveModel(self, model: Any) -> None:
        """
        Saves the model by serialising the model object.
        :param model: Any
        :return:
            - None
        """
        _, exist = self.config.checkPath(self.config.model_dir)
        if not exist:
            os.makedirs(self.config.model_dir)
        path, _ = self.config.checkPath(f"{self.config.model_dir}\\{self.config.dataset_name}", ".model")

        logging.info(f"Saving file '{path}'")
        pickle.dump(model, open(path, "wb"))
