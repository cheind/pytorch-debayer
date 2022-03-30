import logging
from typing import Tuple, List
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import debayer

_logger = logging.getLogger("debayer")


def read_image(
    path: str,
    bayer: bool,
    layout: debayer.Layout = debayer.Layout.RGGB,
    loglevel: int = logging.INFO,
) -> Tuple[np.ndarray, np.ndarray]:
    x: np.ndarray = np.asarray(Image.open(path))
    if x.ndim > 2 and not bayer:
        _logger.log(
            loglevel,
            "Loading multi-channel input as RGB image. "
            "Use `--bayer` to force Bayer interpretation and `--layout` "
            "to specify its layout.",
        )
        _logger.log(loglevel, f"Converting RGB to Bayer image with layout {layout}.")
        # Consider full color, convert to bayer
        b = debayer.utils.rgb_to_bayer(x[..., :3], layout=layout)
    elif x.ndim > 2 and bayer:
        _logger.log(
            loglevel,
            f"Loading multi-channel input as Bayer {layout} image. "
            "Omit `--bayer` to force RGB interpretation. ",
        )
        b = x[..., 0].copy()
    else:
        b = x.copy()
        _logger.log(
            loglevel,
            f"Interpreting single-channel input as Bayer {layout} image. "
            "Use `--layout` to specify its layout.",
        )
    return x, b


def opencv_conversion_code(layout: debayer.Layout):
    import cv2

    return {
        layout.RGGB: cv2.COLOR_BAYER_BG2RGB,
        layout.BGGR: cv2.COLOR_BAYER_RG2RGB,
        layout.GBRG: cv2.COLOR_BAYER_GR2RGB,
        layout.GRBG: cv2.COLOR_BAYER_GB2RGB,
    }[layout]


def create_mosaic(
    imgs: List[np.ndarray], roi: Tuple[int, int, int, int], labels: List[str]
):
    width = roi[1] - roi[0]
    height = roi[3] - roi[2]

    N = len(imgs)
    W, H = plt.figaspect(height / width)

    fig = plt.figure(constrained_layout=True, figsize=(W * len(imgs), H), frameon=False)
    nrow = 1
    ncol = N
    spec = gridspec.GridSpec(
        nrow,
        ncol,
        wspace=0.0,
        hspace=0.0,
        top=1.0 - 0.5 / (nrow + 1),
        bottom=0.5 / (nrow + 1),
        left=0.5 / (ncol + 1),
        right=1 - 0.5 / (ncol + 1),
        figure=fig,
    )

    def plot_image(idx, row, col, shareax=None):
        cmap = "gray" if imgs[idx].ndim == 2 else None
        ax = fig.add_subplot(spec[row, col], sharey=shareax, sharex=shareax)
        ax.imshow(imgs[idx], interpolation="nearest", origin="upper", cmap=cmap)
        ax.set_title(labels[idx], size=10, y=1.0, pad=-14, color="white")
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[3], roi[2])
        return ax

    ax = None
    for idx in range(N):
        ax = plot_image(idx, 0, idx, shareax=ax)
    return fig
