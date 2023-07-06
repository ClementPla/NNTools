from setuptools import setup

setup(
    name='NNTools',
    version='0.1',
    packages=['nntools', 'nntools.nnet', 'nntools.utils', 'nntools.dataset', 'nntools.utils',
              'nntools.tracker'],
    url='https://github.com/ClementPla/NNTools',
    license='MIT',
    author='Clement Playout',
    download_url='https://github.com/ClementPla/NNTools/archive/refs/tags/v_01.tar.gz',
    keywords=['pytorch', 'neural network', 'deep learning', 'training', 'dataset', 'image', 'configuration'],
    author_email='clement.playout@polymtl.ca',
    install_requires=[
        'torch','numpy','matplotlib','tqdm','pyyaml','pandas', 'torchvision', 'opencv-python',
        'segmentation_models_pytorch', 'bokeh', 'pprint'],
    description='Small library built to facilitate the:q training of neural network with Pytorch. '
)
