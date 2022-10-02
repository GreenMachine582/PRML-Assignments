from __future__ import annotations

import os

import examples
import machine_learning as ml

# Constants
local_dir = os.path.dirname(__file__)


def main(dir_: str = local_dir) -> None:
    """
    Gives the user a choice between tasks or datasets.

    :param dir_: Project's path directory, should be a str
    :return: None
    """

    config = ml.Config(dir_, 'BTC-USD')

    dataset = ml.Dataset(config.dataset)
    if not dataset.load():
        raise Exception("Failed to load dataset")

    use_best = True
    while True:
        print(f"""
        0 - Back
        1 - Use best param (Toggle) - {use_best}
        1 - Find Estimator Params
        2 - Find Classifier Params
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
                examples.find_params.findEstimatorParams(dataset, config)
            elif choice == 2:
                examples.find_params.findClassifierParams(dataset, config)
            else:
                print("\nPlease enter a valid choice!")


if __name__ == '__main__':
    main()
