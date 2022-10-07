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
    dataset = examples.processDataset(dataset)

    use_best = True
    while True:
        print(f"""
        0 - Back
        1 - Use best param (Toggle) - {use_best}
        2 - Find Estimator Params
        3 - Find Classifier Params
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
                use_best = not use_best
            elif choice == 2:
                examples.compare_params.findEstimatorParams(deepcopy(dataset), config)
            elif choice == 3:
                examples.compare_params.findClassifierParams(deepcopy(dataset), config)
            else:
                print("\nPlease enter a valid choice!")
