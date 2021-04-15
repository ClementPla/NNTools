import ntpath
import os

import cv2
import yaml


def read_image(filepath, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(filepath, flag)
    if image.ndim == 3:
        return image[:, :, ::-1]  # Change from BGR to RGB
    else:
        return image


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_file


def save_yaml(yaml_file, filepath):
    with open(filepath, 'w') as outfile:
        yaml.dump(yaml_file, outfile, default_flow_style=False)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files.sort(key=lambda x: os.path.getmtime(x))
    if files:
        return files[-1]
