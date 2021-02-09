from nntools.utils import load_yaml



class Config:
    def __init__(self, path):
        self.config_dict = load_yaml(path)
        self.config_path = path

    def __getitem__(self, item):
        return self.config_dict[item]

    def get_path(self):
        return self.config_path

    def __setitem__(self, key, value):
        self.config_dict[key] = value

    def __getattr__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            super(Config, self).__getattr__(item)