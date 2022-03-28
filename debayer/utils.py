import numpy as np

from .debayer import Layouts


def to_bayer(x: np.ndarray, layout=Layouts.RGGB) -> np.ndarray:
    """Converts an RGB image to Bayer RGGB image."""
    H, W, C = x.shape
    p = np.array(layout.value).reshape(2, 2)
    pp = np.tile(p, (H // 2, W // 2))
    b = np.take_along_axis(x, np.expand_dims(pp, -1), -1).squeeze(-1)
    return b
