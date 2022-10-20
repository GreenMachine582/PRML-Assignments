from __future__ import annotations

from copy import deepcopy

import examples
import machine_learning as ml


def getProject(dir_: str, name: str) -> tuple:
    """

    :param dir_:
    :param name:
    :return: config, dataset - tuple[Config, Dataset]
    """
    # TODO: Documentation
    config = ml.Config(dir_, name)

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")
    dataset = examples.processDataset(dataset, overwrite=False)
    return config, dataset


def main(dir_: str) -> None:
    """
    Gives the user a choice between tasks or datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """
    config, dataset = getProject(dir_, 'BTC-USD')

    while True:
        print(f"""
        0 - Back
        1 - Process Dataset (Includes EDA)
        2 - Compare Estimators
        3 - Compare Classifiers
        4 - Compare Params
        """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                return
            elif choice == 1:
                examples.process.main(dir_)
            elif choice == 2:
                examples.compare_models.compareEstimators(deepcopy(dataset), config)
            elif choice == 3:
                examples.compare_models.compareClassifiers(deepcopy(dataset), config)
            elif choice == 4:
                examples.compare_params.compareParams(deepcopy(dataset), config)
            else:
                print("\nPlease enter a valid choice!")
