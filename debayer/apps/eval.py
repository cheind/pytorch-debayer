"""Evaluates debayer algorithms on single file or folder.

The images are assumed to be multi-channel RGB images.

"""

import argparse
import logging
from collections.abc import Callable
from multiprocessing.sharedctypes import Value
from pathlib import Path
import numpy as np
import cv2

import debayer
import matplotlib.pyplot as plt
import torch

from . import utils
from .benchmark import ALL_METHODS

_logger = logging.getLogger("debayer")

IMAGE_SUFFIXES = {".png", ".bmp", ".tif"}
LAYOUT = debayer.Layout.RGGB


def glob_image_paths(path: Path):
    if path.is_file():
        files = [Path(path).resolve()]
    elif path.is_dir():
        files = [
            p.resolve()
            for p in Path(path).glob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
    else:
        raise ValueError("Input path is neither file nor folder.")
    return sorted(files)


def run_opencv(bayer: torch.tensor):
    # assume 8-bit depth here
    bayer = bayer.squeeze().cpu().to(torch.float32).numpy() * 255.0
    bayer = bayer.astype(np.uint8)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB)
    return torch.tensor(rgb).permute(2, 1, 0).to(torch.float32) / 255.0


def reconstruct(
    path: Path,
    methods: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    dev: torch.device,
    prec: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reconstruct image given by path using given methods.
    Returned images are fp32 in range [0..peak], where peak
    is the maximum positive representable value given the image
    bit depth.
    """

    orig, bayer = utils.read_image(
        path, bayer=False, layout=LAYOUT, loglevel=logging.DEBUG
    )
    peak = (1 << orig.dtype.itemsize * 8) - 1

    if orig.ndim != 3 or orig.shape[2] != 3:
        raise ValueError(f"Expected RGB image {path}")

    bayer = (torch.tensor(bayer).to(prec).unsqueeze(0).unsqueeze(0).to(dev)) / peak
    recs = [
        m(bayer).squeeze().permute(1, 2, 0).cpu().to(torch.float32).numpy() * peak
        for m in methods.values()
    ]
    return orig.astype(np.float32), recs, peak


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        default=["Debayer3x3", "Debayer5x5", "OpenCV"],
        nargs="+",
        help="Which methods to run. List of methods or `all`.",
        choices=ALL_METHODS + ["all"],
    )
    parser.add_argument("--half", action="store_true", help="Use 16bit fp precision")
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("path", type=Path, help="Image file or folder")
    args = parser.parse_args()

    # Glob files
    image_paths = glob_image_paths(args.path)
    if len(image_paths) == 0:
        _logger.warning("No image files found")
        return

    # Setup precision and algorithms
    prec = torch.float16 if args.half else torch.float32
    methods = {
        "Debayer2x2": debayer.Debayer2x2(layout=LAYOUT).to(args.dev).to(prec),
        "Debayer3x3": debayer.Debayer3x3(layout=LAYOUT).to(args.dev).to(prec),
        "DebayerSplit": debayer.DebayerSplit(layout=LAYOUT).to(args.dev).to(prec),
        "Debayer5x5": debayer.Debayer5x5(layout=LAYOUT).to(args.dev).to(prec),
        "OpenCV": run_opencv,
    }
    methods = {k: v for k, v in methods.items() if k in args.methods}

    for path in image_paths:
        _logger.info(f"Processing {path}")
        orig, rec, peak = reconstruct(path, methods, args.dev, prec)
        print(peak)


if __name__ == "__main__":
    main()
