from __future__ import annotations

import logging
import os
import sys
from time import time

import examples

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
    logging.info('Starting program')
    while True:
        print("""
        0 - Quit
        1 - Assignment 1 B
        2 - Assignment 2 A
        3 - Assignment 3 A
        """)
        choice = input("Which option number: ")
        try:
            choice = int(choice)
        except ValueError:
            print('\nPlease enter a valid response!')
            choice = None

        if choice is not None:
            if choice == 0:
                break
            elif choice == 1:
                examples.assignment_1_b.main(ROOT_DIR)
            elif choice == 2:
                examples.assignment_2_a.main(ROOT_DIR)
            elif choice == 3:
                examples.assignment_2_a.main(ROOT_DIR)
            else:
                print("\nPlease enter a valid choice!")
    quit_program()


if __name__ == '__main__':
    main()
