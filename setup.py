from setuptools import setup

setup(
    name = 'NNTools',
    version = '0.1.0',
    packages = ['nntools'],
    package_dir={'nntools': 'src/nntools'},
    url='https://github.com/ClementPla/NNTools',
    license='MIT',
    author='Clement Playout',
    author_email = 'clement.playout@polymtl.ca',
    download_url = 'https://github.com/ClementPla/NNTools/archive/refs/tags/v0.1.0.tar.gz',
    keywords = ['PyTorch', 'Neural Network', 'CNN', 'deep learning', 'training', 'dataset', 'image', 'configuration'],
    install_requires = [
        'torch', 'numpy','matplotlib','tqdm','pyyaml','pandas', 'torchvision', 'opencv-python-headless',
        'segmentation_models_pytorch', 'bokeh'],
    description='Light library built to facilitate the training of neural network with Pytorch. '
)
