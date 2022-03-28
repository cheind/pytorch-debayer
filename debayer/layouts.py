import enum


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts."""

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


"""Paddings to convert from layout to RGGB.

Format is left,right,top,bottom paddings

The algorithms in pytorch-debayer use RGGB per default.
In order to support other layouts, the algorithms perform
additional padding.
"""
LayoutPaddingLUT = {
    Layout.RGGB: (0, 0, 0, 0),
    Layout.BGGR: (1, 0, 1, 0),
    Layout.GBRG: (0, 0, 1, 0),
    Layout.GRBG: (1, 0, 0, 0),
}
