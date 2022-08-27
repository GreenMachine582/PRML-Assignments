from __future__ import annotations

import logging
import os
import sys
from time import time

import machine_learning


logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w',
                    format="%(asctime)s - %(levelname)s - '%(message)s' - %(funcName)s")

# Constants
ROOT_DIR = os.path.dirname(__file__)
START_TIME = time()


def quit_program() -> None:
    """
    Closes python in a safe manner.
    :return:
        - None
    """
    logging.info("Exiting program --- %s seconds ---" % round(time() - START_TIME, 2))
    sys.exit(0)


def main() -> None:
    """
    Gives the user a choice between tasks or datasets.
    :return:
        - None
    """
    run = True
    while run:
        try:
            print("""
            0 - Quit
            1 - Assignment 1 Part A 'TBA'
            2 - Assignment 1 Part B 'FashionMNIST'
            """)
            choice = int(input("Which question number: "))
            if choice == 0:
                return
            elif choice == 1:
                pass
            elif choice == 2:
                machine_learning.main('Fashion-MNIST')
                #machine_learning.main(f"{ROOT_DIR}\\datasets\\fashion-mnist_test.csv")
        except ValueError:
            print("Please enter a valid choice!")


if __name__ == '__main__':
    try:
        logging.info('Starting program')
        main()
    except Exception as e:
        logging.error(e)
    raise quit_program()
