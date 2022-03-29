from typing import Union

import numpy as np
import torch

from .layouts import Layout


def rgb_to_bayer(
    x: Union[np.ndarray, torch.Tensor], layout=Layout.RGGB
) -> Union[np.ndarray, torch.Tensor]:
    """Converts an RGB image to Bayer image.

    Args:
        x: (H,W,C) RGB image
        layout: Layout to encode in

    Returns
        b: (H,W) Bayer image with expected layout
    """
    if torch.is_tensor(x):
        C, H, W = x.shape
        p = torch.tensor(layout.value).reshape(2, 2)
        return torch.gather(x, 1, p.repeat(1, H // 2, W // 2))
    else:
        H, W, C = x.shape
        p = np.array(layout.value).reshape(2, 2)
        pp = np.tile(p, (H // 2, W // 2))
        b = np.take_along_axis(x, np.expand_dims(pp, -1), -1).squeeze(-1)
    return b
