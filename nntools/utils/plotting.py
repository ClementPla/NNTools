import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import make_grid, draw_segmentation_masks


def plt_cmap(confMap, cmap_name='RdYlGn'):
    n_classes = confMap.shape[0]
    cmap = cm.get_cmap(cmap_name)
    cmap_r = cm.get_cmap(cmap_name + '_r')

    n_cfmap_row = confMap / confMap.sum(1, keepdims=True)
    n_cfmap_col = confMap / confMap.sum(0, keepdims=True)

    diag_cfmap = cmap(n_cfmap_row)

    colors = cmap_r(n_cfmap_row)
    for i in range(n_classes):
        colors[i, i, :] = diag_cfmap[i, i, :]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((2 * n_classes, 2 * n_classes))
    ax.imshow(colors)
    for i in range(n_classes):
        for j in range(n_classes):

            row = np.nan_to_num(n_cfmap_row[i, j])
            col = np.nan_to_num(n_cfmap_col[i, j])
            if i == j:
                color_row = cmap(row)
                color_col = cmap(col)
            else:
                color_row = cmap_r(row)
                color_col = cmap_r(col)

            ax.text(j, i - 0.25, f'{col:.0%}', color=color_row,
                    horizontalalignment='center',
                    fontweight='black',
                    backgroundcolor=(1.0, 1.0, 1.0, 0.5))

            if confMap[i, j] > 9999:
                ax.text(j, i, f"{confMap[i, j]:.1e}", color=color_row,
                        horizontalalignment='center',
                        fontweight='black',
                        backgroundcolor=(1.0, 1.0, 1.0, 0.95))
            else:
                ax.text(j, i, confMap[i, j], color=color_row,
                        horizontalalignment='center',
                        fontweight='black',
                        backgroundcolor=(1.0, 1.0, 1.0, 0.95))

            ax.text(j, i + 0.25, f'{row:.0%}', color=color_col,
                    horizontalalignment='center',
                    fontweight='black',
                    backgroundcolor=(1.0, 1.0, 1.0, 0.5))

    ax.set_xticks(np.arange(0, n_classes, 1))
    ax.set_yticks(np.arange(0, n_classes, 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, n_classes + 1, 1))
    ax.set_yticklabels(np.arange(1, n_classes + 1, 1))
    ax.set_xticks(np.arange(-.5, n_classes - 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_classes - 1, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    return fig


def create_mosaic(images, masks=None, alpha=0.8, colors=None):
    mosaic_imgs = make_grid(images, normalize=True, nrow=4) * 255
    mosaic_imgs = mosaic_imgs.type(torch.uint8).cpu()
    if masks is not None:
        mosaic_gt = make_grid(masks, normalize=False, nrow=4).cpu()
        mosaic_imgs = draw_segmentation_masks(mosaic_imgs, mosaic_gt.bool(), colors=colors, alpha=alpha)

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
    fig.set_size_inches(fig_size * 10, (10 // col) * row * fig_size)

    for i in range(row):
        for j in range(col):
            ax[i][j].set_axis_off()

            if j + i * col >= len(arrays):
                ax[i][j].imshow(np.zeros_like(np.squeeze(arr)), interpolation='none')
            else:
                name, arr = arrays[j + i * col]

                ax[i][j].set_title(name)
                if arr.ndim == 3 and arr.shape[-1] != 3:
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
