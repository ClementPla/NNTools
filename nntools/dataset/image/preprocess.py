import cv2
import numpy as np

from nntools.dataset.decorators import double_kwarg, single_kwarg
from nntools.utils.sampler import sample


class ImageTransform:
    """
    Assume imgs are of shape hxwxc or hxw
    """
    crop_size = (256, 256)
    cval = 0
    rotation_angle = [-20, 20]
    scale_factor = [0.75, 1.5]
    flags = cv2.INTER_LINEAR
    pad = False
    pad_mode = 'reflect'
    mean = [0, 0, 0]
    std = [1, 1, 1]
    mask_interpolation = cv2.INTER_NEAREST
    img_interpolation = cv2.INTER_LINEAR

    @classmethod
    @single_kwarg
    def normalize(cls, img, mean=None, std=None, **kwargs):
        mean = cls.mean if mean is None else mean
        std = cls.std if std is None else std
        mean = np.asarray(mean)[np.newaxis, np.newaxis, :].astype(np.float32)
        std = np.asarray(std)[np.newaxis, np.newaxis, :].astype(np.float32)
        return (img - mean) / std

    @classmethod
    @double_kwarg
    def random_crop(cls, img, mask=None, pad=None, pad_mode=None, cval=None, crop_size=None, **kwargs):
        """
        :param imgs: Single image or list of images (usually a couple image/groundtruth)
        :param pad_mode: padding mode for handling borders (see numpy.pad).
        It can also be a list of pad modes, following the order in imgs (if list)
        :param pad: boolean, whether or not to use padding (allows cropping more often the border of the image)
        :param cval: in case of constant padding, value used in the borders
        :param crop_size:
        :return:
        """

        crop_size = cls.crop_size if crop_size is None else crop_size
        pad = cls.pad if pad is None else pad
        pad_mode = cls.pad_mode if pad_mode is None else pad_mode
        cval = cls.cval if cval is None else cval
        hcrop, wcrop = crop_size[0], crop_size[1]

        if pad:
            pad_margins = [(hcrop // 2, hcrop // 2), (wcrop // 2, wcrop // 2)]
            kwargs = {'mode': pad_mode}
            if pad_mode == 'constant':
                kwargs['constant_values'] = cval

            if img.ndim == 2:
                img = np.pad(img, pad_margins, **kwargs)
            elif img.ndim == 3:
                img = np.pad(img, pad_margins + [(0, 0)], **kwargs)
            if mask is not None:
                mask = np.pad(mask, pad_margins, **kwargs)

        h, w = img.shape[:2]
        center_w = int(sample([wcrop // 2, w - (wcrop // 2)]))
        center_h = int(sample([hcrop // 2, h - (hcrop // 2)]))
        img = img[center_h - (hcrop // 2):center_h + (hcrop // 2),
              center_w - (wcrop // 2):center_w + (wcrop // 2)]
        if mask is not None:
            mask = mask[center_h - (hcrop // 2):center_h + (hcrop // 2),
                   center_w - (wcrop // 2):center_w + (wcrop // 2)]
            return img, mask

        return img

    @classmethod
    @double_kwarg
    def horizontal_flip(cls, img, mask=None, **kwargs):
        p = sample((0, 1))
        img = img[:, ::-1].copy() if p else img
        if mask is not None:
            mask = mask[:, ::-1].copy() if p else mask
            return img, mask
        return img

    @classmethod
    @double_kwarg
    def vertical_flip(cls, img, mask=None, **kwargs):
        p = sample((0, 1))
        img = img[::-1, :].copy() if p else img
        if mask is not None:
            mask = mask[::-1, :].copy() if p else mask
            return img, mask
        return img

    @classmethod
    @double_kwarg
    def random_rotation(cls, img, mask=None, cval=None, **kwargs):
        angle = sample(list(cls.rotation_angle))
        c = cls.cval if cval is None else cval
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        img = cv2.warpAffine(img, M=rot_mat, dsize=img.shape[1::-1], flags=cls.img_interpolation, borderValue=c)
        if mask is not None:
            mask = cv2.warpAffine(mask, M=rot_mat, dsize=img.shape[1::-1], flags=cls.img_interpolation, borderValue=c)
            return img, mask
        else:
            return img

    @classmethod
    @double_kwarg
    def random_scale(cls, img, mask=None, cval=None, **kwargs):
        f = sample(list(cls.scale_factor))
        cval = cls.cval if cval is None else cval

        def padding(old, new):
            pad = old - new
            if pad % 2 == 0:
                pad1 = pad // 2
                pad2 = pad // 2
            else:
                pad1 = pad // 2
                pad2 = pad // 2 + 1
            return pad1, pad2

        h, w = img.shape[0:2]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cls.img_interpolation)
        if mask is not None:
            mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cls.mask_interpolation)
        new_h, new_w = img.shape[0:2]
        if f < 1:
            padh1, padh2 = padding(h, new_h)
            padw1, padw2 = padding(w, new_w)
            if img.ndim == 2:
                img = np.pad(img, pad_width=[(padh1, padh2), (padw1, padw2)], constant_values=cls.cval)
            else:
                img = np.pad(img, pad_width=[(padh1, padh2), (padw1, padw2), (0, 0)], constant_values=cls.cval)
            if mask is not None:
                mask = np.pad(mask, pad_width=[(padh1, padh2), (padw1, padw2)], constant_values=cls.cval)
        if f >= 1:
            if mask is None:
                img = cls.random_crop(img=img, pad=False, cval=cval, crop_size=(h, w))
            else:
                img, mask = cls.random_crop(img=img, mask=mask, pad=False, cval=cval, crop_size=(h, w))
        if mask is None:
            return img
        else:
            return img, mask

    @classmethod
    @double_kwarg
    def resize(cls, img, mask=None, keep_size_ratio=True, shape=(512, 512)):
        if keep_size_ratio:
            h, w = img.shape[:2]
            max_size = max(h, w)
            scale = shape / max_size
            shape = (int(scale * w), int(scale * h))
        else:
            if isinstance(shape, int):
                shape = (shape, shape)
            else:
                shape = tuple(shape)
        img = cv2.resize(img, dsize=shape, interpolation=cls.img_interpolation)
        if mask is not None:
            mask = cv2.resize(mask, dsize=shape, interpolation=cls.mask_interpolation)
            return img, mask
        return img
