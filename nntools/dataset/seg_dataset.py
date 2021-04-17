import glob
import os

import cv2
import numpy as np
import torch
import tqdm

from nntools.dataset.image_tools import resize
from nntools.tracker import Log
from nntools.utils.io import read_image, path_leaf
from nntools.utils.misc import to_iterable

supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
from .image_dataset import ImageDataset


class SegmentationDataset(ImageDataset):
    def __init__(self, img_url,
                 mask_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 n_classes=None,
                 sort_function=None, use_cache=False):
        self.path_masks = to_iterable(mask_url)
        if self.path_masks == '':
            self.path_masks = None
        self.use_masks = self.path_masks is not None
        self.cmap_name = 'jet_r'
        self.n_classes = n_classes

        super(SegmentationDataset, self).__init__(img_url, shape, keep_size_ratio, recursive_loading, sort_function, use_cache)

    def get_class_count(self):
        from .utils import get_segmentation_class_count
        return get_segmentation_class_count(self)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for path in self.path_img:
                self.img_filepath.extend(glob.glob(path + prefix + extension, recursive=recursive))
            if self.use_masks:
                for path in self.path_masks:
                    self.gts.extend(glob.glob(path + prefix + extension, recursive=recursive))

        self.img_filepath = np.asarray(self.img_filepath)
        if self.use_masks:
            self.gts = np.asarray(self.gts)

        """
        Sorting files
        """
        img_filenames = [path_leaf(path).split('.')[0] for path in self.img_filepath]
        mask_filenames = [path_leaf(path).split('.')[0] for path in self.gts]
        if self.use_masks and len(img_filenames) != len(mask_filenames):
            Log.warn("Mismatch between the number of image (%i) and masks (%i) found!" % (
                len(img_filenames), len(mask_filenames)))

        if self.sort_function is None:
            img_argsort = np.argsort(self.img_filepath)
            self.img_filepath = self.img_filepath[img_argsort]
            if self.use_masks:
                mask_argsort = np.argsort(self.gts)
                self.gts = self.gts[mask_argsort]
        else:
            sort_key_img = np.argsort([self.sort_function(x) for x in img_filenames])
            self.img_filepath = self.img_filepath[sort_key_img]

            if self.use_masks:
                sort_key_mask = np.argsort([self.sort_function(x) for x in mask_filenames])
                self.gts = self.gts[sort_key_mask]

    def multiply(self, factor):
        """
        Artificially increase the size of the dataset by a given multiplication factor
        :param factor:
        :return:
        """
        decimal = factor - int(factor)
        factor = int(factor)
        rest = int(decimal * len(self))

        self.img_filepath = np.repeat(self.img_filepath, factor)
        self.gts = np.repeat(self.gts, factor)
        if rest:
            self.img_filepath = np.concatenate((self.img_filepath, self.img_filepath[:rest]))
            self.gts = np.concatenate((self.gts, self.gts[:rest]))

    def load_array(self, item):
        if self.use_cache:
            return self.read_sharred_array(item)
        else:
            img = self.load_image(item)
            if self.use_masks:
                filepath = self.gts[item]
                mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
                mask = resize(image=mask, shape=self.shape, keep_size_ratio=self.keep_size_ratio, flag=cv2.INTER_NEAREST)
                return img, mask
            return img

    def __getitem__(self, item):

        arrays = self.load_array(item)
        if self.use_masks:
            img, mask = arrays
        else:
            img = arrays

        if self.composer:
            if self.use_masks:
                img, mask = self.composer(image=img, mask=mask)
            else:
                img = self.composer(image=img)

        img = self.transpose_img(img)

        output = (torch.from_numpy(img),)
        if self.use_masks:
            output = output + (torch.from_numpy(mask).long(),)
        if self.return_indices:
            output = output + (item,)
        return output

    def cache(self):
        self.use_cache = False
        if self.use_masks:
            self.sharred_imgs, self.sharred_masks = self.init_cache()
        else:
            self.sharred_imgs = self.init_cache()[0]

        print('Caching dataset...')
        for i in tqdm.tqdm(range(1, len(self))):
            if self.use_masks:
                img, mask = self.load_array(i)
                self.sharred_imgs[i] = img
                self.sharred_masks[i] = mask
            else:
                img = self.load_array(i)
                self.sharred_imgs[i] = img

        self.use_cache = True

    def get_mask(self, item):
        filepath = self.gts[item]
        mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
        mask = resize(image=mask, shape=self.shape, keep_size_ratio=self.keep_size_ratio, flag=cv2.INTER_NEAREST)
        if self.composer:
            mask = self.composer(mask=mask)
        if self.return_indices:
            return mask, item
        else:
            return mask

    def standardize(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def plot(self, item, show=True, save=False, save_folder='tmp/', classes=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm

        plt.rcParams['image.cmap'] = 'gray'

        if self.use_masks:
            img, mask = self[item]
            img = img.numpy()
            img = self.standardize(img)
            mask = mask.numpy()
            mask = np.squeeze(mask)
            n_classes = self.n_classes if self.n_classes is not None else np.max(mask) + 1
            cmap = cm.get_cmap(self.cmap_name, n_classes)

            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(18, 8)
            ax[0].imshow(np.squeeze(img.transpose((1, 2, 0))))
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
            img = self[item][0]
            img = img.numpy()
            img = self.standardize(img)
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(8, 8)
            ax.imshow(np.squeeze(img.transpose((1, 2, 0))))
            ax.set_axis_off()
        fig.tight_layout()
        if show:
            fig.show()
        if save:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            filename = os.path.basename(self.img_filepath[item])
            fig.savefig(os.path.join(save_folder, filename))
            plt.close(fig)
