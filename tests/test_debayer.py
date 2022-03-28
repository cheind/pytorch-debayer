import numpy as np
import torch
from itertools import product
import pytest

from debayer.debayer import Debayer2x2, Debayer3x3, Debayer5x5, Layout
from debayer.utils import to_bayer

Colors = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]


def _bayer_to_torch(x: np.ndarray, dtype=torch.float32, dev: str = "cpu"):
    return torch.tensor(x).to(dtype).unsqueeze(0).unsqueeze(0).to(dev)


@pytest.mark.parametrize("layout", Layout)
@pytest.mark.parametrize("color", Colors)
@pytest.mark.parametrize("klass", [Debayer2x2, Debayer3x3, Debayer5x5])
def test_monochromatic_images(layout, color, klass):
    """Algorithms should be able to reconstruct ideal monochromatic bayer images without error."""
    rgb = np.tile(
        np.array(color, dtype=np.float32).reshape(1, 1, -1),
        (6, 8, 1),
    )
    b = _bayer_to_torch(to_bayer(rgb, layout=layout))
    r = klass(layout=layout)(b)

    # import matplotlib.pyplot as plt
    # plt.imshow(r.squeeze().permute(1, 2, 0).cpu().to(torch.float32).numpy())
    # plt.show()
    assert r.shape == (1, 3, 6, 8)
    assert (r == torch.tensor(color).view(1, -1, 1, 1)).all()