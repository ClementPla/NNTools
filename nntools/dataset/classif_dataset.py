from .abstract_dataset import ImageDataset
import glob
import numpy as np
import os
supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
import torch


class ClassificationDataset(ImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 map_class=None,
                 label_present=True):
        self.map_class = map_class
        self.label_present = label_present

        super(ClassificationDataset, self).__init__(img_url, shape, keep_size_ratio, recursive_loading)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            self.img_filepath.extend(glob.glob(self.path_img + prefix + extension, recursive=recursive))

        if self.label_present:
            for f in self.img_filepath:
                self.gts.append(os.path.basename(os.path.dirname(f)))

            unique_labels = np.unique(self.gts)
            self.n_classes = len(unique_labels)
            self.gts = np.asarray(self.gts)

            if self.map_class is None:
                self.map_class = {unique_labels[i]: i for i in np.arange(len(unique_labels))}
            for k, v in self.map_class.items():
                self.gts[self.gts == k] = v
        self.gts = self.gts.astype(int)
        self.img_filepath = np.asarray(self.img_filepath)

    def __getitem__(self, item):
        img = self.load_image(item)
        kwargs = {'image': img}
        if self.composer:
            img = self.composer(**kwargs)

        img = self.transpose_img(img)

        output = (torch.from_numpy(img),)
        if self.label_present:
            output += (torch.tensor(self.gts[item], dtype=torch.long),)
        if self.return_indices:
            output += (item, )
        return output

    def get_class_count(self):
        from .utils import get_classification_class_count
        return get_classification_class_count(self)

    def plot(self, item, show=True, save=False, save_folder='tmp/'):
        import matplotlib.pyplot as plt

        plt.rcParams['image.cmap'] = 'gray'
        img, l = self[item][:2]
        img = img.numpy()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        ax.imshow(np.squeeze(img.transpose((1, 2, 0))))
        ax.set_axis_off()
        fig.tight_layout()
        label = list(self.map_class.keys())[list(self.map_class.values()).index(int(l))]
        ax.set_title("Class %s (%i)" % (label, l))

        if show:
            fig.show()
        if save:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            filename = os.path.basename(self.img_filepath[item])
            fig.savefig(os.path.join(save_folder, filename))
            plt.close(fig)