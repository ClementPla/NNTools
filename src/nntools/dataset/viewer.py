import logging
import math
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from nntools.utils.plotting import plot_images


def convert_dict_to_plottable(dict_arrays):
    plotted_arrays = {}
    for k, v in dict_arrays.items():
        if isinstance(v, torch.Tensor):
            v = v.numpy()
            if v.ndim == 3:
                v = v.transpose((1, 2, 0))
        plotted_arrays[k] = v
    return plotted_arrays


class Viewer:
    def __init__(self, dataset) -> None:
        self.d = dataset
        self.cmap_name = "jet_r"

    def plot(self, item: int, classes: Optional[List[str]] = None, fig_size: int = 1):
        arrays = self.d.__getitem__(item, return_indices=False)
        arrays = convert_dict_to_plottable(arrays)
        plot_images(arrays, self.cmap_name, classes=classes, fig_size=fig_size)

    def get_mosaic(
        self,
        n_items: int = 9,
        shuffle: bool = False,
        indexes: Optional[List[int]] = None,
        resolution: Tuple[int, int] = (512, 512),
        show: bool = False,
        fig_size: int = 1,
        save: Optional[bool] = None,
        add_labels: bool = False,
        n_row: Optional[int] = None,
        n_col: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        if indexes is None:
            if shuffle:
                indexes = np.random.randint(0, len(self.d), n_items)
            else:
                indexes = np.arange(n_items)

        ref_dict = self.d.__getitem__(0, return_indices=False, return_tag=False)
        ref_dict = convert_dict_to_plottable(ref_dict)
        count_images = 0
        for k, v in ref_dict.items():
            if isinstance(v, np.ndarray) and not np.isscalar(v):
                count_images += 1
        if n_row is None and n_col is None:
            n_row = math.ceil(math.sqrt(n_items))
            n_col = math.ceil(n_items / n_row)
        elif n_row is None:
            n_row = math.ceil(n_items / n_col)
        elif n_col is None:
            n_col = math.ceil(n_items / n_row)
        if n_row * n_col < n_items:
            logging.warning(
                "With %i columns, %i row(s), only %i items can be plotted"
                % (n_col, n_row, n_row * n_col)
            )
            n_items = n_row * n_col
        pad = 50 if add_labels else 0
        cols = []

        for r in range(n_row):
            row = []
            for c in range(n_col):
                i = n_row * c + r
                if i >= n_items:
                    for n in range(count_images):
                        tmp = np.zeros((resolution[0] + pad, resolution[1], 3))
                        row.append(tmp)
                    continue
                index = indexes[i]
                data = self.d.__getitem__(index, return_indices=False, return_tag=False)
                data = convert_dict_to_plottable(data)

                for k, v in data.items():
                    if v.ndim == 3 and v.shape[-1] != 3:
                        v_tmp = np.argmax(v, axis=-1) + 1
                        v_tmp[v.max(axis=-1) == 0] = 0
                        v = v_tmp
                    if v.ndim == 3:
                        v = (v - v.min()) / (v.max() - v.min())
                    if v.ndim == 2:
                        n_classes = np.max(v) + 1 if n_classes is None else n_classes
                        if n_classes == 1:
                            n_classes = 2
                        cmap = plt.get_cmap(self.cmap_name, n_classes)
                        v = cmap(v)[:, :, :3]
                    if v.shape:
                        v = cv2.resize(v, resolution, cv2.INTER_NEAREST_EXACT)
                    if add_labels and v.shape:
                        v = np.pad(v, ((pad, 0), (0, 0), (0, 0)))
                        text = ""
                        for k in data.keys():
                            if k in self.d.gts:
                                text += " " + str(self.d.gts[k][index])
                            elif k in self.d.img_filepath:
                                text += " " + os.path.basename(
                                    self.d.img_filepath[k][index]
                                )
                        text = os.path.basename(text)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1.0
                        fontColor = (255, 255, 255)
                        lineType = 2

                        textsize = cv2.getTextSize(text, font, fontScale, lineType)[0]
                        textX = (v.shape[1] - textsize[0]) // 2
                        textY = (textsize[1] + pad) // 2

                        bottomLeftCornerOfText = textX, textY
                        cv2.putText(
                            v,
                            text,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType,
                        )
                    if v.shape:
                        row.append(v)

            rows = np.hstack(row)

            cols.append(rows)

        mosaic = np.vstack(cols)
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(mosaic, interpolation="none")
            fig.set_size_inches(
                fig_size * 5 * count_images * n_col, 5 * n_row * fig_size
            )
            plt.axis("off")
            plt.tight_layout()
            fig.show()
        if save:
            assert isinstance(save, str)
            cv2.imwrite(save, (mosaic * 255)[:, :, ::-1])

        return mosaic
