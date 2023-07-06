# NNTools
A set of tools to facilitate the development of Neural Networks with Pytorch.

While it has initially an ambition to framework such as [pytorch-lightning](https://lightning.ai/docs/pytorch/latest/) or [Ignite](https://pytorch.org/ignite/), it became very quickly obvious that a new framework would not bring any value.

Therefore, NNTools has been skinned off. It is now mostly a set of classes/functions to handle boilerplate code regarding:
 * Dataset of images, either for classification, segmentation or other usecases. It handles composition of (preprocessing) functions and can be interface with [Albumentation](https://albumentations.ai/docs/)
 * Plotting tools, a set of convenient functions for some standards plotting with [Bokeh](https://bokeh.org/)
 * Configuration files. 


## Handling of data

NNTools provides the `dataset` module to automatically create datasets (as subclasses of Pytorch's [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html))

