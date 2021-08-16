import os
from collections import OrderedDict
from nntools.utils.io import load_yaml
import pprint

TAG_EXPAND = '*'
TAG_IGNORE = '^'


class DoubleDict:
    def __init__(self, ordered=True):
        if ordered:
            self._dict = OrderedDict()  # This dict may contains keys with tags such as * and !
            self._parsed_dict = OrderedDict()  # This  dict is a duplicata of _dict, but tag free
        else:
            self._dict = dict()  # This dict may contains keys with tags such as * and !
            self._parsed_dict = dict()  # This  dict is a duplicata of _dict, but tag free

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

    def update(self, other_dict: dict):
        self._dict.update(other_dict)
        tag_cleaning(self._dict, self._parsed_dict)

    def clean_dict(self):
        filter_dict = dict()
        tag_parsing(self._dict, filter_dict)
        return filter_dict

    def __repr__(self):
        return str(self._parsed_dict)

    def items(self):
        return self._dict.items()

    @property
    def parsed_dict(self):
        return self._parsed_dict


class Config:
    def __init__(self, path=None):
        super(Config, self).__init__()

        self.keys_dict = DoubleDict(ordered=False)

        if path is not None:
            self.load_yaml(path)

    def get_path(self):
        return self.config_path

    def load_yaml(self, path):
        config_dict = load_yaml(path)
        recursive_dict_replacement(config_dict)
        self.keys_dict.update(config_dict)
        self.config_path = os.path.realpath(path)

    @property
    def tracked_params(self):
        return self.keys_dict.clean_dict()

    def __getitem__(self, item):
        return self.keys_dict[item]

    def __setitem__(self, key, value):
        self.keys_dict[key] = value

    def __repr__(self):
        return pprint.pformat(self.keys_dict.parsed_dict)


def recursive_dict_replacement(org_dict):
    for k, v in org_dict.items():
        if isinstance(v, dict):
            recursive_dict_replacement(v)
            new_dict = DoubleDict(ordered=False)
            new_dict.update(v)
            org_dict[k] = new_dict


def tag_cleaning(original_dict, new_dict):
    for k, v in original_dict.items():
        if isinstance(v, dict) or isinstance(v, DoubleDict) or isinstance(v, OrderedDict):
            new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)] = dict()
            tag_cleaning(v, new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)])
        else:
            new_dict[k.strip(TAG_EXPAND+TAG_IGNORE)] = v


def tag_parsing(original_dict, new_dict, parent_key='', level=0):
    for k, v in original_dict.items():
        if k.startswith(TAG_IGNORE):
            continue
        else:
            if (isinstance(v, dict) or isinstance(v, DoubleDict) or isinstance(v, OrderedDict)) and (k.endswith(TAG_EXPAND) or level == 0):
                k = k.strip(TAG_EXPAND+TAG_IGNORE)
                tag_parsing(v, new_dict, '%s/%s' % (parent_key, k) if parent_key else k, level+1)
            else:
                k = k.strip(TAG_EXPAND+TAG_IGNORE)
                key = '%s/%s'%(parent_key, k) if parent_key else k
                new_dict[key] = v


if __name__ == '__main__':
    path = '../../tests/c_file_test.yaml'
    c = Config(path)
    c['Loss']['type'] = 'A'
    # pprint.pprint(c, compact=False)
    print(c)

