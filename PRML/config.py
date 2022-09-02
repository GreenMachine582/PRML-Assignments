from __future__ import annotations

import json
import logging
import os
from pathlib import Path


class Config(object):
    """
    Config stores default and important values that the MachineLearning
    class can use. The end-user can save, load and update the attributes.
    """

    def __init__(self, working_dir: str, dataset_name: str):
        self.working_dir = working_dir
        self.dataset_name = dataset_name

        self.config_dir = working_dir + '\\configs'
        self.dataset_dir = working_dir + '\\datasets'
        self.model_dir = working_dir + '\\models'

        self.show_figs = True
        self.show_small_responses = True

        # dataset related
        self.seperator = ','
        self.target = 'target'
        self.names = None

        # training related
        self.split_ratio = 0.8
        self.random_seed = 0
        self.model_type = 'LogisticRegression'

        self.load() if os.path.isfile(self.config_dir.format(self.dataset_name)) else self.save()

    @staticmethod
    def checkPath(path: str, extension: str = '') -> tuple:
        """
        Adds an extension if not already included in path, then checks if path exists.
        :param path: str
        :param extension: str
        :return:
            - path, exist - tuple[str]
        """
        if os.path.splitext(path)[1] != extension:
            path = path + extension
        exist = True if os.path.exists(path) else False
        return path, exist

    def update(self, kwargs: dict) -> None:
        """
        Updates the class attributes with given keys and values.
        :param kwargs: dict[str: Any]
        :return:
            - None
        """
        logging.info('Updating class attributes')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self) -> None:
        """
        Attempts to load the config file using json, if file is not found a warning
        will be logged.
        :return:
            - None
        """
        path, exist = self.checkPath(f'{self.config_dir}\\{self.dataset_name}', '.json')

        if exist:
            logging.info(f'Loading config {path}')
            with open(path, 'r', encoding='utf-8') as f:
                self.update(json.load(f))
        else:
            logging.warning(f"Missing file '{path}'")

    def save(self) -> None:
        """
        Saves the config attributes as an indented dict in a json file, to allow
        end-users to edit and easily view the default configs.
        :return:
            - None
        """
        _, exist = self.checkPath(self.config_dir)
        if not exist:
            os.makedirs(self.config_dir)
        path, _ = self.checkPath(f'{self.config_dir}\\{self.dataset_name}', '.json')

        logging.info(f"Saving file '{path}'")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)
