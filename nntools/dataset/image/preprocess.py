import cv2
import numpy as np

from nntools.dataset.decorators import double_kwarg, single_kwarg
from nntools.utils.sampler import sample


# class ImageTransform:
#     """
#     Assume imgs are of shape hxwxc or hxw
#     """
#     crop_size = (256, 256)
#     cval = 0
#     rotation_angle = [-20, 20]
#     scale_factor = [0.75, 1.5]
#     flags = cv2.INTER_LINEAR
#     pad = False
#     pad_mode = 'reflect'
#     mean = [0, 0, 0]
#     std = [1, 1, 1]
#     mask_interpolation = cv2.INTER_NEAREST
#     img_interpolation = cv2.INTER_LINEAR

@single_kwarg
def normalize(cls, img, mean=None, std=None, **kwargs):
    mean = cls.mean if mean is None else mean
    std = cls.std if std is None else std
    mean = np.asarray(mean)[np.newaxis, np.newaxis, :].astype(np.float32)
    std = np.asarray(std)[np.newaxis, np.newaxis, :].astype(np.float32)
    return (img - mean) / std


@double_kwarg
def random_crop(img, crop_size, mask=None, pad=False, pad_mode='reflect', cval=0, **kwargs):
    """
    :param imgs: Single image or list of images (usually a couple image/groundtruth)
    :param pad_mode: padding mode for handling borders (see numpy.pad).
    It can also be a list of pad modes, following the order in imgs (if list)
    :param pad: boolean, whether or not to use padding (allows cropping more often the border of the image)
    :param cval: in case of constant padding, value used in the borders
    :param crop_size:
    :return:
    """

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


@double_kwarg
def horizontal_flip(img, mask=None, **kwargs):
    img = img[:, ::-1].copy()
    if mask is not None:
        mask = mask[:, ::-1].copy()
        return img, mask
    return img


@double_kwarg
def vertical_flip(img, mask=None, **kwargs):
    img = img[::-1, :].copy()
    if mask is not None:
        mask = mask[::-1, :].copy()
        return img, mask
    return img


@double_kwarg
def random_rotation(img, rotation_angle=None, mask=None, flag=cv2.INTER_LINEAR, cval=0):
    angle = sample(list(rotation_angle))
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img = cv2.warpAffine(img, M=rot_mat, dsize=img.shape[1::-1], flags=flag, borderValue=cval)
    if mask is not None:
        mask = cv2.warpAffine(mask, M=rot_mat, dsize=img.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue=cval)
        return img, mask
    else:
        return img


@double_kwarg
def random_scale(img, scale_factor=None, mask=None, cval=0, flag=cv2.INTER_LINEAR):
    f = sample(list(scale_factor))

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
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=flag)
    if mask is not None:
        mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    new_h, new_w = img.shape[0:2]
    if f < 1:
        padh1, padh2 = padding(h, new_h)
        padw1, padw2 = padding(w, new_w)
        if img.ndim == 2:
            img = np.pad(img, pad_width=[(padh1, padh2), (padw1, padw2)], constant_values=cval)
        else:
            img = np.pad(img, pad_width=[(padh1, padh2), (padw1, padw2), (0, 0)], constant_values=cval)
        if mask is not None:
            mask = np.pad(mask, pad_width=[(padh1, padh2), (padw1, padw2)], constant_values=cval)
    if f >= 1:
        if mask is None:
            img = random_crop(img=img, pad=False, cval=cval, crop_size=(h, w))
        else:
            img, mask = random_crop(img=img, mask=mask, pad=False, cval=cval, crop_size=(h, w))
    if mask is None:
        return img
    else:
        return img, mask


@double_kwarg
def resize(img, keep_size_ratio=True, shape=(512, 512), flag=cv2.INTER_LINEAR):
    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        shape = tuple(shape)
    if keep_size_ratio:
        h, w = img.shape[:2]
        max_size = np.max(img.shape[:2])
        amax_size = np.argmax(img.shape[:2])
        scale = shape[amax_size] / max_size
        shape = (int(scale * w), int(scale * h))

    img = cv2.resize(img, dsize=shape, interpolation=flag)
    return img
