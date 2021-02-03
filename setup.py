from setuptools import setup

setup(
    name='NNTools',
    version='0.0',
    packages=['nntools', 'nntools.nnet', 'nntools.nnet.ops', 'nntools.nnet.models', 'nntools.utils', 'nntools.dataset',
              'nntools.tracker'],
    url='',
    license='',
    author='Clement Playout',
    author_email='clement.playout@polymtl.ca',
    description='Small library built to facilitate the training of neural network with Pytorch. '
)
