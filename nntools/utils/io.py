import cv2
import yaml


def load_image(filepath, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(filepath, flag)
    if image.ndim == 3:
        return image[:, :, ::-1]  # Change from BGR to RGB
    else:
        return image


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config(config, filepath):
    with open(filepath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
