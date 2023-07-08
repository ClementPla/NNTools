# NNTools
A set of tools to facilitate the development of Neural Networks with Pytorch.

While it has initially an ambition to framework such as [pytorch-lightning](https://lightning.ai/docs/pytorch/latest/) or [Ignite](https://pytorch.org/ignite/), it became very quickly obvious that a new framework would not bring any value.

Therefore, NNTools has been skinned off. It is now mostly a set of classes/functions to handle boilerplate code regarding:
 * Dataset of images, either for classification, segmentation or other usecases. It handles composition of (preprocessing) functions and can be interface with [Albumentation](https://albumentations.ai/docs/)
 * Plotting tools, a set of convenient functions for some standards plotting with [Bokeh](https://bokeh.org/)
 * Configuration files. 


## Handling of data

NNTools provides the `dataset` module to automatically create datasets (as subclasses of Pytorch's [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html))

```python
import nntools.dataset as D
folder = ...
dataset = D.ImageDataset(folder, shape=(1024, 1024), keep_size_ratio=True, recursive_loading=True)

folders = {'image':folder1, 'preprocessing':folder2}
dataset = D.MultiImageDataset(folders, shape=(1024, 1024), keep_size_ratio=True, recursive_loading=True)


label_filepath = 'label.csv' # or label.xls
dataset = D.ClassificationDataset(folder, shape=(1024, 1024), 
                                  label_filepath=label_filepath,
                                  file_column='name', gt_column='retinopathy',
                                  keep_size_ratio=True, recursive_loading=True)

mask1 = 'mask1/'
mask2 = 'mask2/'
masks_url = {'Class1':mask1, 'Class2':mask2}
dataset = D.SegmentationDataset(folder, shape=(1024, 1024), mask_url=masks_url, keep_size_ratio=True, recursive_loading=True)

```

## Composition logic for preprocessing on the fly

```python
import albumentations as A

@D.nntools_wrapper
def merge_labels_to_multiclass(Class1, Class2):
    mask = np.zeros_like(Vessels)
    mask[Class1 == 1] = 1
    mask[Class2 == 1] = 2
    return {'mask':mask}


composer = D.Composition()
composer.add(merge_labels_to_multiclass)
data_aug = A.Compose([A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)]) 
composer.add(data_aug) 
dataset.composer = composer
```

## Configuration

Write your configuration in a usual [Yaml format](https://yaml.org/)

`config.yaml`
```yaml
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
conf = Config('config.yaml')

# conf['model']['architecture']=='resnet'
```
The `Config` class provides two small utilities functions:

    1. `config.tracked_params` will return all the parameters in a uploadable format.
    2. The format of the tracked params can be adjusted using `*` and `^` markers. 
Check the [example notebook](/notebooks/Config.ipynb) for more details.

