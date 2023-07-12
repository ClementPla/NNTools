import os
import pprint
from collections import OrderedDict

from nntools.utils.io import load_yaml

TAG_COMPRESS = "*"
TAG_IGNORE = "^"


class DictParsed(dict):
    def __init__(self, other_dict=None):
        self.keys_with_tags = dict()
        if other_dict is None:
            super(DictParsed, self).__init__()
        else:
            super(DictParsed, self).__init__(self.parse_other_dict(other_dict))

    def update(self, __m, **kwargs) -> None:
        __m = self.parse_other_dict(__m)
        super(DictParsed, self).update(__m, **kwargs)

    def parse_other_dict(self, other):
        for k in other.copy().keys():
            tags = (k.startswith(TAG_IGNORE), k.endswith(TAG_COMPRESS))
            new_key = k.strip(TAG_COMPRESS + TAG_IGNORE)
            if any(tags):
                self.keys_with_tags[new_key] = tags
                other[new_key] = other.pop(k)
        return other

    def __getitem__(self, item):
        return super(DictParsed, self).__getitem__(item.strip(TAG_COMPRESS + TAG_IGNORE))

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = DictParsed(value)
        tags = (key.startswith(TAG_IGNORE), key.endswith(TAG_COMPRESS))
        new_key = key.strip(TAG_COMPRESS + TAG_IGNORE)
        if any(tags):
            self.keys_with_tags[new_key] = tags
        super(DictParsed, self).__setitem__(new_key, value)

    def tracked_params(self, parent="", level=0):
        tracked_params = OrderedDict()
        for k, v in self.items():
            tags = self.keys_with_tags.get(k, (False, False))
            if tags[0]:
                continue
            current_key = "%s%s" % (parent, k)
            if isinstance(v, DictParsed):
                if not tags[1] or (level == 0):
                    tmp_t_params = v.tracked_params(current_key + "/", level + 1)
                    tracked_params.update(tmp_t_params)
                else:
                    tracked_params[current_key] = v.filtered_dict()
            else:
                tracked_params[current_key] = v
        return tracked_params

    def filtered_dict(self):
        filtered_dict = dict()
        for k in self.keys():
            tags = self.keys_with_tags.get(k, (False, False))
            if tags[0]:
                continue
            else:
                filtered_dict[k] = self.get(k)
        return filtered_dict


class Config:
    def __init__(self, path=None):
        super(Config, self).__init__()

        self.keys_dict = DictParsed()
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
        return self.keys_dict.tracked_params()

    def __getitem__(self, item):
        return self.keys_dict[item]

    def __setitem__(self, key, value):
        self.keys_dict[key] = value

    def __repr__(self):
        return pprint.pformat(self.keys_dict)

    def get(self, key, default=None):
        return self.keys_dict.get(key, default)

    def pop(self, key, default=None):
        return self.keys_dict.pop(key, default)


def recursive_dict_replacement(org_dict):
    for k, v in org_dict.items():
        if isinstance(v, dict):
            recursive_dict_replacement(v)
            new_dict = DictParsed()
            new_dict.update(v)
            org_dict[k] = new_dict
