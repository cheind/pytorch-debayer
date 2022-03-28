import torch
import numpy as np

from .layouts import Layout


def to_bayer(x: np.ndarray, layout=Layout.RGGB) -> np.ndarray:
    """Converts an RGB image to Bayer RGGB image.

    Args:
        x: (H,W,C) RGB image
        layout: Layout to encode in

    Returns
        b: (H,W) Bayer image with expected layout
    """
    H, W, C = x.shape
    p = np.array(layout.value).reshape(2, 2)
    pp = np.tile(p, (H // 2, W // 2))
    b = np.take_along_axis(x, np.expand_dims(pp, -1), -1).squeeze(-1)
    return b


def remove_padding(x: torch.Tensor, paddings: tuple[int, int, int, int]):
    l, r, t, b = paddings
    h, w = x.shape[-2:]
    return x[..., t : h - b, l : w - r]
