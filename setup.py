from setuptools import setup

setup(
    name = 'NNTools',
    version = '0.1.0',
    packages = ['nntools'],
    url='https://github.com/ClementPla/NNTools',
    license='MIT',
    author='Clement Playout',
    author_email = 'clement.playout@polymtl.ca',
    download_url = 'https://github.com/ClementPla/NNTools/archive/refs/tags/v0.1.0.tar.gz',
    keywords = ['PyTorch', 'Neural Network', 'CNN', 'deep learning', 'training', 'dataset', 'image', 'configuration'],
    author_email = 'clement.playout@polymtl.ca',
    install_requires = [
        'torch', 'numpy','matplotlib','tqdm','pyyaml','pandas', 'torchvision', 'opencv-python',
        'segmentation_models_pytorch', 'bokeh'],
    description='Light library built to facilitate the training of neural network with Pytorch. '
)
