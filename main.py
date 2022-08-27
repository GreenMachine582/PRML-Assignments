from __future__ import annotations

import logging
import os
import sys
from time import time

import PRML

# Sets up the in-built logger to record key information and save it to a text file
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w',
                    format="%(asctime)s - %(levelname)s - '%(message)s' - %(funcName)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # Outputs the loggings into screen output

# Constants
ROOT_DIR = os.path.dirname(__file__)
START_TIME = time()


def quit_program() -> None:
    """
    Closes python in a safe manner.
    :return:
        - None
    """
    logging.info("Exiting program - %s seconds -" % round(time() - START_TIME, 2))
    sys.exit(0)


def main() -> None:
    """
    Gives the user a choice between tasks or datasets.
    :return:
        - None
    """
    config = PRML.Config(ROOT_DIR)
    run = True
    while run:
        try:
            print("""
            0 - Quit
            1 - 'fashion-mnist_test.csv'
            2 - 'fashion-mnist_train.csv'
            3 - 'Fashion-MNIST'
            4 - 'iris.data.csv'
            """)
            choice = int(input("Which question number: "))
            if choice == 0:
                return
            elif choice == 1:
                config.target = 'label'
                ml = PRML.MachineLearning(config, 'fashion-mnist_test.csv')
                ml.main()
                return
            elif choice == 2:
                config.target = 'label'
                ml = PRML.MachineLearning(config, 'fashion-mnist_train.csv')
                ml.main()
                return
            elif choice == 3:
                ml = PRML.MachineLearning(config, 'Fashion-MNIST')
                ml.main()
                return
            elif choice == 4:
                config.target = 'class'
                config.names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
                ml = PRML.MachineLearning(config, 'iris.data.csv')
                ml.main()
                return
        except ValueError:
            print("Please enter a valid choice!")


if __name__ == '__main__':
    logging.info('Starting program')
    main()
    raise quit_program()
