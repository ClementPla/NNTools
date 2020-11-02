from nntools.dataset.image import SegmentationDataset, preprocess
from nntools.dataset.utils import get_class_count, class_weighting


if __name__ == '__main__':
    from nntools.dataset.image.tools import Composition, DataAugment
    root_img = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    root_gt = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    dataset = SegmentationDataset(root_img, root_gt, shape=(800, 800))

    import albumentations as A

    img_path = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    gt_path = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    dataset = SegmentationDataset(img_path, gt_path, shape=(1500, 1500))

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)])
    from nntools.dataset.image.tools import Composition, DataAugment

    composer = Composition()
    composer << DataAugment(random_rotate=True, ratio=0.5).auto_init() << aug
    dataset.set_composition(composer)
    dataset.plot(0)


