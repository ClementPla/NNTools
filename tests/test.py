import albumentations as A

from nntools import NN_FILL_UPSAMPLE
import nntools.dataset as D
import cv2
import numpy as np

@D.nntools_wrapper
def merge_masks(image, mask, lesion):
    return {'image': image, 'mask': mask}


@D.nntools_wrapper
def fundus_autocrop(image):
    r_img = image[:, :, 0]
    _, mask = cv2.threshold(r_img, 255, 1, cv2.THRESH_BINARY)

    not_null_pixels = cv2.findNonZero(mask)
    if not_null_pixels is None:
        return {'image': image, 'mask': mask.astype(np.float32)}
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {'image': image, 'mask': mask.astype(np.float32)}
    return {'image': image[y_range[0]:y_range[1], x_range[0]:x_range[1]],
            'mask': mask[y_range[0]:y_range[1], x_range[0]:x_range[1]].astype(np.float32)}

if __name__ == '__main__':
    imgs = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    masks = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    test = '/home/clement/Documents/phd/DR/MessidorAnnotation/Test/'

    inputs_masks = {'mask': masks, 'lesion': masks}

    dataset = D.SegmentationDataset(imgs, inputs_masks, (1024, 764),
                                    keep_size_ratio=True,
                                    filling_strategy=NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={'lesion': 'mask'})
    composer << fundus_autocrop
    dataset.set_composition(composer)
    dataset.plot(0)
    print(dataset[0]['image'].shape)
    print(dataset[0]['mask'].shape)