import albumentations as A

from nntools import NN_FILL_UPSAMPLE
import nntools.dataset as D

@D.nntools_wrapper
def merge_masks(image, mask, lesion):
    return {'image': image, 'mask': mask}


if __name__ == '__main__':
    imgs = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    masks = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    test = '/home/clement/Documents/phd/DR/MessidorAnnotation/Test/'

    inputs_masks = {'mask': masks, 'lesion': masks}

    dataset = D.SegmentationDataset(imgs, inputs_masks, (512, 764),
                                    keep_size_ratio=True,
                                    filling_strategy=NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={'lesion': 'mask'})
    composer
    dataset.set_composition(composer)
    dataset.plot(0)
    print(dataset[0]['image'].shape)