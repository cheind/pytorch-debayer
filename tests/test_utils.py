import numpy as np

import debayer


def test_to_bayer():
    rgb = np.random.rand(20, 30, 3)
    b = debayer.utils.rgb_to_bayer(rgb, layout=debayer.Layout.RGGB)
    assert b.shape == (20, 30)
    assert b[0, 0] == rgb[0, 0, 0]
    assert b[0, 1] == rgb[0, 1, 1]
    assert b[0, 2] == rgb[0, 2, 0]
    assert b[1, 2] == rgb[1, 2, 1]
    assert b[1, 3] == rgb[1, 3, 2]

    b = debayer.utils.rgb_to_bayer(rgb, layout=debayer.Layout.BGGR)
    assert b.shape == (20, 30)
    assert b[0, 0] == rgb[0, 0, 2]
    assert b[0, 1] == rgb[0, 1, 1]
    assert b[0, 2] == rgb[0, 2, 2]
    assert b[1, 2] == rgb[1, 2, 1]
    assert b[1, 3] == rgb[1, 3, 0]

    b = debayer.utils.rgb_to_bayer(rgb, layout=debayer.Layout.GBRG)
    assert b.shape == (20, 30)
    assert b[0, 0] == rgb[0, 0, 1]
    assert b[0, 1] == rgb[0, 1, 2]
    assert b[0, 2] == rgb[0, 2, 1]
    assert b[1, 2] == rgb[1, 2, 0]
    assert b[1, 3] == rgb[1, 3, 1]

    b = debayer.utils.rgb_to_bayer(rgb, layout=debayer.Layout.GRBG)
    assert b.shape == (20, 30)
    assert b[0, 0] == rgb[0, 0, 1]
    assert b[0, 1] == rgb[0, 1, 0]
    assert b[0, 2] == rgb[0, 2, 1]
    assert b[1, 2] == rgb[1, 2, 2]
    assert b[1, 3] == rgb[1, 3, 1]
