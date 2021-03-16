from nntools.utils.io import load_yaml
import os

class Config(dict):
    def __init__(self, path):
        super(Config, self).__init__()
        config_dict = load_yaml(path)
        self.update(config_dict)
        self.config_path = os.path.realpath(path)

    def get_path(self):
        return self.config_path

    def get_item(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default


