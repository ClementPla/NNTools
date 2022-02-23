import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import make_grid, draw_segmentation_masks
import torch


def create_mosaic(images, masks=None):
    mosaic_imgs = make_grid(images, normalize=True) * 255
    mosaic_imgs = mosaic_imgs.type(torch.uint8)
    if masks is not None:
        mosaic_gt = make_grid(masks, normalize=False)
        mosaic_imgs = draw_segmentation_masks(mosaic_imgs, mosaic_gt.bool())

    return mosaic_imgs


def plot_images(arrays_dict, cmap_name='jet_r', classes=None, fig_size=1):
    arrays = [(k, v) for k, v in arrays_dict.items() if isinstance(v, np.ndarray)]
    nb_plots = len(arrays)
    if nb_plots == 1:
        row, col = 1, 1
    else:
        row, col = int(math.ceil(nb_plots / 2)), 2

    fig, ax = plt.subplots(row, col)
    if row == 1:
        ax = [ax]
    if col == 1:
        ax = [ax]
    fig.set_size_inches(fig_size * 10, (10//col) * row * fig_size)

    for i in range(row):
        for j in range(col):
            ax[i][j].set_axis_off()

            if j + i * col >= len(arrays):
                ax[i][j].imshow(np.zeros_like(np.squeeze(arr)), interpolation='none')
            else:
                name, arr = arrays[j + i * col]

                ax[i][j].set_title(name)
                if arr.ndim == 3 and arr.shape[-1] > 3:
                    arr_tmp = np.argmax(arr, axis=-1) + 1
                    arr_tmp[arr.max(axis=-1) == 0] = 0
                    arr = arr_tmp
                if arr.ndim == 3:
                    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
                    ax[i][j].imshow(np.squeeze(arr), cmap='gray', interpolation='none')
                elif arr.ndim == 2:
                    n_classes = np.max(arr) + 1
                    if n_classes == 1:
                        n_classes = 2
                    cmap = cm.get_cmap(cmap_name, n_classes)
                    ax[i][j].imshow(np.squeeze(arr), cmap=cmap, interpolation='none')
                    divider = make_axes_locatable(ax[i][j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cax.imshow(np.expand_dims(np.arange(n_classes), 0).transpose((1, 0)), aspect='auto', cmap=cmap)
                    cax.yaxis.set_label_position("right")
                    cax.yaxis.tick_right()
                    if classes is not None:
                        cax.set_yticklabels(labels=classes)
                    cax.yaxis.set_ticks(np.arange(n_classes))
                    cax.get_xaxis().set_visible(False)

    fig.tight_layout()
    fig.show()