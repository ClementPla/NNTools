import os

import cv2
import torch
import yaml

from nntools.utils.const import supportedExtensions


def read_image(filepath, flag=None):
    filepath = str(filepath)
    if flag is None:
        flag = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filepath, flag)
    if image.ndim == 3:
        return image[:, :, ::-1]  # Change from BGR to RGB
    else:
        return image


def save_image(image, filepath, invert_channels=True):
    filepath = str(filepath)
    if image.ndim == 3 and invert_channels:
        image = image[:, :, ::-1]
    success = cv2.imwrite(filepath, image)
    if not success:
        print(image.shape)
        raise ValueError(f"Could not save image at {filepath}, cv2 returned  {success}")


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_file


def save_yaml(yaml_file, filepath):
    with open(filepath, "w") as outfile:
        yaml.dump(yaml_file, outfile, default_flow_style=False)


def path_leaf(filepath: str):
    """
    Return the filename (without the extension) from a filepath
    Args:
        path (str): /abc/def/test.jpeg

    Returns:
        str: test
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def path_folder_leaf(filepath: str):
    """
    Return the container folder of the given file (from a filepath)

    Args:
        filepath (str): /abc/def/test.jpeg

    Returns:
        str: def
    """
    return os.path.dirname(filepath).split("/")[-1]


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath, filtername=None):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files.sort(key=lambda x: os.path.getmtime(x))
    if filtername is not None:
        files = [f for f in files if filtername in os.path.basename(f)]
    if files:
        return files[-1]


def jit_load(project_folder, experiment, run_name, run_id, filename=None, filtername="best"):
    folder_path = os.path.join(project_folder, experiment, run_name, "trained_model", run_id)
    script_path = os.path.join(folder_path, "model_scripted.pth")
    if not os.path.exists(script_path):
        return ValueError("No scripted model found")
    model = torch.jit.load(script_path)

    if filename:
        path = os.path.join(folder_path, filename)
    else:
        path = folder_path

    model.load(path, load_most_recent=filename is None, filtername=filtername)
    return model


def list_files_in_folder(folder, recursive=True):
    files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(folder):
            for f in filenames:
                if os.path.splitext(f)[1] in supportedExtensions:
                    files.append(os.path.join(dirpath, f))
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1] in supportedExtensions]

    return files
