from __future__ import annotations

from copy import deepcopy

import examples
import machine_learning as ml


def main(dir_: str) -> None:
    """
    Gives the user a choice between tasks or datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """
    config = ml.Config(dir_, 'BTC-USD')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")

    run = True
    while run:
        print("""
        0 - Back
        1 - Process Dataset (Includes EDA)
        2 - Compare Estimators
        3 - Compare Classifiers
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
                examples.process.main(deepcopy(dataset))
            elif choice == 2:
                examples.compare_models.compareEstimators(deepcopy(dataset), config.random_state)
            elif choice == 3:
                examples.compare_models.compareClassifiers(deepcopy(dataset), config.random_state)
            else:
                print("\nPlease enter a valid choice!")
