from __future__ import annotations

import logging
import os

from pandas import DataFrame
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import utils


def load(dir_: str, name: str) -> object:
    """
    Loads the model from a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :return: model - object
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    name = utils.joinPath(name, ext='.model')
    model = utils.load(dir_, name)
    if model is None:
        logging.warning(f"Failed to load model '{name}'")
    return model


def save(dir_: str, name: str, model: object) -> bool:
    """
    Saves the model to a model file.

    :param dir_: Directory path of file, should be a str
    :param name: Name of file, should be a str
    :param model: The classifier or estimator, should be an object
    :return: completed - bool
    """
    if not os.path.exists(dir_):
        logging.warning(f"Path '{dir_}' does not exist")
        return False

    name = utils.joinPath(name, ext='.model')
    completed = utils.save(dir_, name, model)
    if not completed:
        logging.warning(f"Failed to save model '{name}'")
    return completed


def gridSearch(model: object, param_grid: dict | list, **kwargs) -> GridSearchCV:
    """
    Exhaustive search over specified parameter values for a model.

    :param model: Model's classifier or estimator, should be an object
    :param param_grid: Enables searching over any sequence of parameter settings, should be a
     dict[str] | list[dict[str: Any]]
    :param kwargs: Additional keywords to pass to GridSearchCV
    :return: cv_results - GridSearchCV
    """
    defaults = {'n_jobs': -1, 'cv': 10, 'verbose': 2, 'return_train_score': True}
    grid_search = GridSearchCV(model, param_grid, **dict(defaults, **kwargs))
    return grid_search


class Model(object):

    def __init__(self, config: dict, **kwargs) -> None:
        """
        Create an instance of Model

        :param config: Model's configurations, should be a dict
        :param kwargs: Additional keywords to pass to update
        :return: None
        """
        self.model: object = None
        self.dir_: str = ''
        self.name: str = ''

        self.update(**dict(config, **kwargs))

    def update(self, **kwargs) -> None:
        """
        Updates the instance attributes, if given attributes are present
        in instance and match existing types.

        :key model: Model's classifier or estimator, should be an object
        :key dir_: Model's path directory, should be a str
        :key name: Model's name, should be a str
        :return: None
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                logging.error(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            else:
                attr_ = getattr(self, key)
                if isinstance(attr_, (type(value), type(None))):
                    setattr(self, key, value)
                else:
                    logging.error(f"'{key}': got '{type(value).__name__}' but expected type is "
                                  f"'{type(attr_).__name__}'")
        logging.info(f"Updated model '{self.name}' attributes")

    def load(self) -> bool:
        """
        Loads the model.

        :return: completed - bool
        """
        model = load(self.dir_, self.name)
        if model is None:
            return False
        self.model = model
        return True

    def save(self) -> bool:
        """
        Saves the model.

        :return: completed - bool
        """
        utils.makePath(self.dir_)
        return save(self.dir_, self.name, self.model)

    def gridSearch(self, param_grid: dict | list, X: DataFrame, y: DataFrame) -> GridSearchCV:
        """
        Grid searches the model then fits with given data.

        :param param_grid: Enables searching over any sequence of parameter settings, should be a
         dict[str] | list[dict[str: Any]]
        :param X: Independent features, should be a DataFrame
        :param y: Dependent variables, should be a DataFrame
        :return: results_cv - GridSearchCV
        """
        cv_results = gridSearch(self.model, param_grid, cv=TimeSeriesSplit(10))
        cv_results.fit(X, y)
        return cv_results
