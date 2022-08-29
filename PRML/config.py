from __future__ import annotations


class Config(object):

    def __init__(self, _dir):
        self.dir = _dir
        self.dataset_dir = _dir + '\\datasets\\'
        self.model_dir = _dir + '\\models\\'

        self.show_figs = True
        self.show_small_responses = False

        self.seperator = ','
        self.target = 'target'
        self.names = None

        self.split_ratio = 0.8
        self.random_seed = 0
        self.model_type = 'LogisticRegression'
