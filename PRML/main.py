from __future__ import annotations

import os
import sys
from time import time

import machine_learning


# Constants
ROOT_DIR = os.path.dirname(__file__)
START_TIME = time()


def quit_program():
    print("--- %s seconds ---" % round(time() - START_TIME, 2))
    sys.exit()


def main():
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
                break
            elif choice == 1:
                pass
            elif choice == 2:
                machine_learning.main('Fashion-MNIST')
        except ValueError:
            print("Please enter a valid choice!")
        except Exception as e:
            print(e)

    quit_program()


if __name__ == '__main__':
    main()
