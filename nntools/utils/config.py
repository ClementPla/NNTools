from nntools.utils.io import load_yaml


class Config:
    def __init__(self, path):
        self.config_dict = load_yaml(path)
        self.config_path = path
        self.verify()

    def __getitem__(self, item):
        return self.config_dict[item]

    def get_path(self):
        return self.config_path

    def __setitem__(self, key, value):
        self.config_dict[key] = value

    def verify(self):
        if not isinstance(self.config_dict['Loss'], dict):
            self.config_dict['Loss'] = {self.config_dict['Loss']: {}}


