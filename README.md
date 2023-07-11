# NNTools
A set of tools to facilitate the development of Neural Networks with Pytorch.

While it initially had the ambition to compete with frameworks such as [pytorch-lightning](https://lightning.ai/docs/pytorch/latest/) or [Ignite](https://pytorch.org/ignite/), it became very quickly obvious that a new framework [would not bring any value](https://xkcd.com/927/).

Therefore, NNTools has been trimmed down. It is now mostly a set of classes/functions to handle boilerplate code regarding:
* Dataset of images, either for classification, segmentation or other usecases. It handles composition of (preprocessing) functions and can be interface with [Albumentations](https://albumentations.ai/docs/)
* Plotting tools, a set of convenient functions for some standards plotting with [Bokeh](https://bokeh.org/)
* Configuration files

## Installation
This package relies on OpenCV, which can be a bit tricky to install and [does not play nice](https://github.com/opencv/opencv-python#installation-and-usage) when different flavors are installed in the same environment. NNTools only requires the non-contrib headless flavor. Since we cannot specify "opt-out" dependencies and the different flavors are not detected to be super-sets of others, if you need some features that are not included in the headless version, or if such a version is already installed in your environment, you will need to uninstall conflicting versions, and then (re)install the version you need:

```bash
pip install opencv-contrib-python  # Say you already have this one
pip install nntools  # Installs opencv-python-headless on top, which conflicts with opencv-contrib-python
pip freeze | grep -E 'opencv(-contrib)?-python(-headless)?' | xargs pip uninstall -y  # Uninstall everything
pip install opencv-contrib-python  # (Re)install the version you need
```

If you are starting with a clean environment and don't need any of the features of the contrib/non-headless versions, you can simply install the package:

```bash
pip install nntools
```

## Handling of data

NNTools provides the `dataset` module to automatically create datasets (as subclasses of Pytorch's [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html))

```python
import nntools.dataset as D


folder = ...
dataset = D.ImageDataset(folder, shape=(1024, 1024), keep_size_ratio=True, recursive_loading=True)

folders = {"image": folder1, "preprocessing": folder2}
dataset = D.MultiImageDataset(folders, shape=(1024, 1024), keep_size_ratio=True, recursive_loading=True)

label_filepath = "label.csv" # or label.xls
dataset = D.ClassificationDataset(
    folder,
    shape=(1024, 1024), 
    label_filepath=label_filepath,
    file_column="name",
    gt_column="retinopathy",
    keep_size_ratio=True,
    recursive_loading=True,
)

mask1 = "mask1/"
mask2 = "mask2/"
masks_url = {"Class1": mask1, "Class2": mask2}
dataset = D.SegmentationDataset(
    folder,
    shape=(1024, 1024),
    mask_url=masks_url,
    keep_size_ratio=True,
    recursive_loading=True,
)
```

## Composition logic for preprocessing on the fly

```python
import albumentations as A


@D.nntools_wrapper
def merge_labels_to_multiclass(Class1, Class2):
    mask = np.zeros_like(Vessels)
    mask[Class1 == 1] = 1
    mask[Class2 == 1] = 2
    return {"mask":mask}


composer = D.Composition()
composer.add(merge_labels_to_multiclass)
data_aug = A.Compose([A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)]) 
composer.add(data_aug) 
dataset.composer = composer
```

## Configuration

Write your configuration in a usual [YAML format](https://yaml.org/)

```yaml
# `config.yaml`
experiment:
    name: MyExperiment
    author: Myself
model:
    architecture: resnet
    layers: 5
training:
    lr: 0.0001
    optimizer:
        type: sgd
        momentum: 0.001

^tracking:
    url: localhost:4200 
```

It can then be opened and used like a regular dict in your Python script:

```python
from nntools.utils import Config


conf = Config("config.yaml")
# conf["model"]["architecture"] == "resnet"
```

The `Config` class provides two small utilities functions:
    1. `config.tracked_params` will return all the parameters in a uploadable format.
    2. The format of the tracked params can be adjusted using `*` and `^` markers. 

Check the [example notebook](/notebooks/Config.ipynb) for more details.
