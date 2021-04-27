import albumentations as A

import nntools.dataset as D


@D.nntools_wrapper
def merge_masks(image, mask, lesion):
    return {'image': image, 'mask': mask}


if __name__ == '__main__':
    imgs = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    masks = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    test = '/home/clement/Documents/phd/DR/MessidorAnnotation/Test/'

    inputs_masks = {'mask': masks, 'lesion': masks}

    dataset = D.SegmentationDataset(imgs, inputs_masks, (512, 512), filling_strategy=D.NN_FILL_UPSAMPLE)
    composer = D.Composition()
    composer << A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={'lesion': 'mask'})
    composer
    dataset.set_composition(composer)
    print(len(dataset))
    dataset.plot(0)

    inputs = {'image': '/home/clement/Images/', 'pair': '/home/clement/Images/'}
    dataset = D.MultiImageDataset(inputs, (512, 512))

    print(len(dataset))

    dataset.plot(0)
    print(dataset[0]['image'].shape)
