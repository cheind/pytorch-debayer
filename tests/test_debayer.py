import numpy as np
import torch
from itertools import product
import pytest

from debayer.debayer import Debayer2x2
from debayer.layouts import Layout
from debayer.utils import to_bayer

Colors = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]


def _bayer_to_torch(x: np.ndarray, dtype=torch.float32, dev: str = "cpu"):
    return torch.tensor(x).to(dtype).unsqueeze(0).unsqueeze(0).to(dev)


@pytest.mark.parametrize("layout", [Layout.GRBG])
@pytest.mark.parametrize("color", [Colors[0]])
# @pytest.mark.parametrize("layout", Layout)
# @pytest.mark.parametrize("color", Colors)
@pytest.mark.parametrize("klass", [Debayer2x2])
def test_monochromatic_images(layout, color, klass):
    rgb = np.tile(
        np.array(color, dtype=np.float32).reshape(1, 1, -1),
        (6, 8, 1),
    )
    b = _bayer_to_torch(to_bayer(rgb, layout=layout))
    r = Debayer2x2(layout=layout)(b)

    import matplotlib.pyplot as plt

    plt.imshow(r.squeeze().permute(1, 2, 0).cpu().to(torch.float32).numpy())
    plt.show()
    assert r.shape == (1, 3, 6, 8)
    assert (r == torch.tensor(color).view(1, -1, 1, 1)).all()

    # def test_debayer2x2():
    # b_rggb = _bayer_to_torch(to_bayer(green, layout=Layout.BGGR))
    # rgb = Debayer2x2()(b_rggb)
    # assert rgb.shape == (1, 3, 6, 8)
    # assert (rgb == torch.tensor([0, 1.0, 0]).view(1, -1, 1, 1)).all()
    # print(rgb[0, 0].min(), rgb[0, 0].max())

    # print(rgb.shape)
    # pass
