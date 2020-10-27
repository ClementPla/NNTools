from nntools.dataset.image import SegmentationDataset, preprocess
from nntools.dataset.utils import get_class_count, class_weighting


if __name__ == '__main__':
    from nntools.dataset.image.tools import Composition, DataAugment
    root_img = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    root_gt = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    dataset = SegmentationDataset(root_img, root_gt, shape=(800, 800))

    transform = Composition()
    transform << DataAugment(ratio=0., vertical_flip=False).auto_init()
    dataset.compose(transform)
    dataset.plot(0)

