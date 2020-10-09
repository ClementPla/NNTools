import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from nntools.dataset.image.preprocess import ImageTransform
from nntools.tracker.warnings import Tracker
from nntools.utils.io import load_image
from nntools.utils.path import path_leaf

fileExtensions = ["jpg", "jpeg", "png", "tiff"]


class SegmentationDataset(Dataset):
    def __init__(self, img_url,
                 mask_url,
                 shape=None,
                 keep_size_ratio=True,
                 recursive_loading=True,
                 n_classes=None):

        super(SegmentationDataset, self).__init__()
        self.path_img = img_url
        self.path_masks = mask_url
        if self.path_masks == '':
            self.path_masks = None
        self.use_masks = self.path_masks is not None
        self.img_filepath = []
        self.mask_filepath = []
        self.list_files(recursive_loading)

        self.composer = None
        self.keep_size_ratio = keep_size_ratio
        self.shape = shape

        self.cmap_name = 'jet_r'
        self.n_classes = n_classes

    def list_files(self, recursive):
        for extension in fileExtensions:
            prefix = "**/*." if recursive else "*."
            self.img_filepath.extend(glob.glob(self.path_img + prefix + extension, recursive=recursive))
            if self.use_masks:
                self.mask_filepath.extend(glob.glob(self.path_masks + prefix + extension, recursive=recursive))

        self.img_filepath = np.asarray(self.img_filepath)
        self.mask_filepath = np.asarray(self.mask_filepath)

        """
        Sorting files
        """
        img_filenames = np.asarray([path_leaf(path).split('.')[0] for path in self.img_filepath])
        mask_filenames = np.asarray([path_leaf(path).split('.')[0] for path in self.mask_filepath])
        if self.use_masks and len(img_filenames) != len(mask_filenames):
            Tracker.warn("Mismatch between the number of images (%i) and masks (%i) found!" % (
                len(img_filenames), len(mask_filenames)))

        img_argsort = np.argsort(img_filenames)

        if self.use_masks:
            self.img_filepath = self.img_filepath[img_argsort][:len(mask_filenames)]
            mask_argsort = np.argsort(mask_filenames)
            self.mask_filepath = self.mask_filepath[mask_argsort][:len(img_filenames)]

    def subset(self, indices):
        self.img_filepath = self.img_filepath[indices]
        self.mask_filepath = self.mask_filepath[indices]

    def __len__(self):
        return len(self.img_filepath)

    def multiply(self, factor):
        decimal = factor - int(factor)
        factor = int(factor)
        rest = int(decimal * len(self))

        self.img_filepath = np.repeat(self.img_filepath, factor)
        self.mask_filepath = np.repeat(self.mask_filepath, factor)
        if rest:
            self.img_filepath = np.concatenate((self.img_filepath, self.img_filepath[:rest]))
            self.mask_filepath = np.concatenate((self.mask_filepath, self.mask_filepath[:rest]))

    def __getitem__(self, item):
        filepath = self.img_filepath[item]
        img = load_image(filepath)
        if self.use_masks:
            filepath = self.mask_filepath[item]
            mask = load_image(filepath, cv2.IMREAD_GRAYSCALE)
            img, mask = ImageTransform.resize(img=img, mask=mask, shape=self.shape,
                                              keep_size_ratio=self.keep_size_ratio)
        else:
            img = ImageTransform.resize(img=img, shape=self.shape,
                                        keep_size_ratio=self.keep_size_ratio)
        img = img.astype(np.float32) / 255.
        kwargs = {'img': img}
        if self.composer:
            if self.use_masks:
                kwargs['mask'] = mask
                img, mask = self.composer(**kwargs)
            else:
                img = self.composer(**kwargs)
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)

        if self.use_masks:
            return torch.from_numpy(img), torch.from_numpy(mask).long()
        else:
            return torch.from_numpy(img)

    def filename(self, items):
        filepaths = self.img_filepath[items]
        if isinstance(filepaths, list):
            return [os.path.basename(f) for f in filepaths]
        else:
            return os.path.basename(filepaths)

    def plot(self, item, show=True, save=False, savefolder='tmp/', classes=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm

        if self.use_masks:
            img, mask = self[item]
            mask = np.squeeze(mask)
            n_classes = self.n_classes if self.n_classes is not None else np.max(mask)
            cmap = cm.get_cmap(self.cmap_name, n_classes)

            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(18, 8)
            ax[0].imshow(img.transpose((1, 2, 0)))
            ax[1].imshow(mask, cmap=cmap)

            ax[0].set_axis_off()
            ax[1].set_axis_off()

            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cax.imshow(np.expand_dims(np.arange(n_classes), 0).transpose((1, 0)), aspect='auto', cmap=cmap)
            cax.yaxis.set_label_position("right")
            cax.yaxis.tick_right()
            if classes is not None:
                cax.set_yticklabels(labels=classes)
            cax.yaxis.set_ticks(np.arange(n_classes))
            cax.get_xaxis().set_visible(False)

        else:
            img = self[item]
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 8)
            ax.imshow(img)
            ax.set_axis_off()
        fig.tight_layout()
        if show:
            fig.show()
        if save:
            if not os.path.exists(savefolder):
                os.mkdir(savefolder)
            filename = os.path.basename(self.img_filepath[item])
            fig.savefig(os.path.join(savefolder, filename))
            plt.close(fig)
