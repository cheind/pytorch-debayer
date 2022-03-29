import enum


class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)
