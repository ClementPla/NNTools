import os
from collections import OrderedDict
from nntools.utils.io import load_yaml
import pprint

TAG_EXPAND = '*'
TAG_IGNORE = '^'
class Config:
    def __init__(self, path=None):
        super(Config, self).__init__()
        self._dict = OrderedDict() # This dict may contains keys with tags such as * and !
        self._parsed_dict = OrderedDict() # This  dict is a duplicata of _dict, tag free
        if path is not None:
            self.load_yaml(path)

    def get_path(self):
        return self.config_path

    def load_yaml(self, path):
        config_dict = load_yaml(path)
        self.update(config_dict)
        tag_cleaning(self._dict, self._parsed_dict)
        self.config_path = os.path.realpath(path)

    def update(self, other_dict: dict):
        self._dict.update(other_dict)

    def get_item(self, key, default=None):
        if key in self._parsed_dict:
            return self._parsed_dict[key]
        else:
            return default

    def __getitem__(self, key):
        return self.get_item(key)

    def __setitem__(self, key, value):
        self._dict[key] = value
        self._parsed_dict[key.strip(TAG_EXPAND+TAG_IGNORE)] = value

    @property
    def tracked_params(self):
        filter_dict = dict()
        tag_parsing(self._dict, filter_dict)
        return filter_dict

    def __repr__(self):
        return pprint.pformat(self._parsed_dict)


def tag_cleaning(original_dict, new_dict):
    for k, v in original_dict.items():
        if isinstance(v, dict):
            new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)] = dict()
            tag_cleaning(v, new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)])
        else:
            new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)] = v


def tag_parsing(original_dict, new_dict, parent_key='', level=0):
    for k, v in original_dict.items():
        if k.startswith(TAG_IGNORE):
            continue
        else:
            if isinstance(v, dict) and (k.endswith(TAG_EXPAND) or level==0):
                k = k.strip(TAG_EXPAND+TAG_IGNORE)
                tag_parsing(v, new_dict, '%s: %s' % (parent_key, k) if parent_key else k, level+1)
            else:
                k = k.strip(TAG_EXPAND+TAG_IGNORE)
                key = '%s: %s'%(parent_key, k) if parent_key else k
                new_dict[key] = v


if __name__ == '__main__':
    path = '../../tests/c_file_test.yaml'
    c = Config(path)
    pprint.pprint(c, compact=False)

