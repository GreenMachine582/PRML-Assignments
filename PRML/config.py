from __future__ import annotations


class Config(object):

    def __init__(self, _dir):
        self.dir = _dir
        self.dataset_dir = f"{_dir}\\datasets\\"

        self.split_ratio = 0.8
        self.random_seed = 0
        self.seperator = ','
        self.model_type = 'LogisticRegression'
        self.names = None
        self.target = 'target'
