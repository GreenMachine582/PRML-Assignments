
import logging
import os

import PRML

# Constants
local_dir = os.path.dirname(__file__)


def main():
    config = PRML.Config(local_dir, 'Fashion-MNIST')
    ml = PRML.MachineLearning(config)
    raw_dataset = ml.loadDataset()

    if raw_dataset is None:
        logging.error("Couldn't load a dataset")

    dataset, X, y = ml.processData(raw_dataset)

    dataset, X, y = ml.extractFeatures(dataset, X, y)

    ml.exploratoryDataAnalysis(dataset, X, y)

    X_train, X_test, y_train, y_test = ml.splitDataset(X, y)

    model = ml.trainModel(X_train, y_train)

    ml.resultAnalysis(model, X, X_test, y_test)

    ml.saveModel(model)
    # model = ml.loadModel()
    # ml.resultAnalysis(model, X, X_test, y_test)


if __name__ == '__main__':
    main()
