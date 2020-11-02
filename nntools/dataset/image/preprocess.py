import cv2
import numpy as np

from nntools.dataset.image.decorators import preprocess
from nntools.utils.random import sample


@preprocess
def normalize(image, mean=None, std=None):
    mean = mean if mean is None else mean
    std = std if std is None else std
    mean = np.asarray(mean)[np.newaxis, np.newaxis, :].astype(np.float32)
    std = np.asarray(std)[np.newaxis, np.newaxis, :].astype(np.float32)
    return (image - mean) / std


@preprocess
def random_crop(image, crop_size, mask=None, pad=False, pad_mode='reflect', cval=0):
    hcrop, wcrop = crop_size[0], crop_size[1]
    if pad:
        pad_margins = [(hcrop // 2, hcrop // 2), (wcrop // 2, wcrop // 2)]
        kwargs = {'mode': pad_mode}
        if pad_mode == 'constant':
            kwargs['constant_values'] = cval

        if image.ndim == 2:
            image = np.pad(image, pad_margins, **kwargs)
        elif image.ndim == 3:
            image = np.pad(image, pad_margins + [(0, 0)], **kwargs)
        if mask is not None:
            mask = np.pad(mask, pad_margins, **kwargs)

    h, w = image.shape[:2]
    center_w = int(sample([wcrop // 2, w - (wcrop // 2)]))
    center_h = int(sample([hcrop // 2, h - (hcrop // 2)]))
    image = image[center_h - (hcrop // 2):center_h + (hcrop // 2),
          center_w - (wcrop // 2):center_w + (wcrop // 2)]
    if mask is not None:
        mask = mask[center_h - (hcrop // 2):center_h + (hcrop // 2),
               center_w - (wcrop // 2):center_w + (wcrop // 2)]
        return image, mask

    return image


@preprocess
def horizontal_flip(image, mask=None):
    image = image[:, ::-1].copy()
    if mask is not None:
        mask = mask[:, ::-1].copy()
        return image, mask
    return image


@preprocess
def vertical_flip(image, mask=None):
    image = image[::-1, :].copy()
    if mask is not None:
        mask = mask[::-1, :].copy()
        return image, mask
    return image


@preprocess
def random_rotation(image, rotation_angle=(-10, 10), mask=None, flag=cv2.INTER_LINEAR, cval=0):
    angle = sample(rotation_angle)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, M=rot_mat, dsize=image.shape[1::-1], flags=flag, borderValue=cval)
    if mask is not None:
        mask = cv2.warpAffine(mask, M=rot_mat, dsize=image.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue=cval)
        return image, mask
    else:
        return image


@preprocess
def random_scale(image, scale_factor=(0.75,1.25), mask=None, cval=0, flag=cv2.INTER_LINEAR, pad_mode='constant'):
    f = sample(list(scale_factor))
    kwargs = {'mode': pad_mode}
    if pad_mode == 'constant':
        kwargs['constant_values'] = cval

    def padding(old, new):
        pad = old - new
        if pad % 2 == 0:
            pad1 = pad // 2
            pad2 = pad // 2
        else:
            pad1 = pad // 2
            pad2 = pad // 2 + 1
        return pad1, pad2

    h, w = image.shape[:2]
    image = cv2.resize(image, None, fx=f, fy=f, interpolation=flag)
    if mask is not None:
        mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    new_h, new_w = image.shape[0:2]
    if f < 1:
        padh1, padh2 = padding(h, new_h)
        padw1, padw2 = padding(w, new_w)
        if image.ndim == 2:
            image = np.pad(image, pad_width=[(padh1, padh2), (padw1, padw2)], **kwargs)
        else:
            image = np.pad(image, pad_width=[(padh1, padh2), (padw1, padw2), (0, 0)], **kwargs)
        if mask is not None:
            mask = np.pad(mask, pad_width=[(padh1, padh2), (padw1, padw2)], **kwargs)
    if f >= 1:
        if mask is None:
            image = random_crop(image=image, pad=False, cval=cval, crop_size=(h, w))
        else:
            image, mask = random_crop(image=image, mask=mask, pad=False, cval=cval, crop_size=(h, w))
    if mask is None:
        return image
    else:
        return image, mask


@preprocess
def resize(image, keep_size_ratio=True, shape=(512, 512), flag=cv2.INTER_LINEAR):
    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        shape = tuple(shape)
    if keep_size_ratio:
        h, w = image.shape[:2]
        max_size = np.max(image.shape[:2])
        amax_size = np.argmax(image.shape[:2])
        scale = shape[amax_size] / max_size
        shape = (int(scale * w), int(scale * h))

    image = cv2.resize(image, dsize=shape, interpolation=flag)
    return image

