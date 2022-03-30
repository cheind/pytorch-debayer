"""Evaluates debayer algorithms on single file or folder.

The images are assumed to be multi-channel RGB images.

"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import debayer
import numpy as np
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
    bayer = bayer.squeeze().cpu().to(torch.float32).numpy() * 255.0
    bayer = bayer.astype(np.uint8)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB)
    return torch.tensor(rgb).permute(2, 0, 1).to(torch.float32) / 255.0


def reconstruct(
    path: Path,
    methods: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    dev: torch.device,
    prec: torch.dtype,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    """Reconstruct image given by path using given methods and return PSNR values."""

    orig, bayer = utils.read_image(
        path, bayer=False, layout=LAYOUT, loglevel=logging.DEBUG
    )
    bpp = orig.dtype.itemsize * 8
    max_intensity = (2 ** bpp) - 1

    if orig.ndim != 3 or orig.shape[2] != 3:
        raise ValueError(f"Expected RGB image {path}")

    # Reconstruct from Bayer
    bayer = (
        torch.tensor(bayer).to(prec).unsqueeze(0).unsqueeze(0).to(dev)
    ) / max_intensity
    recs = [m(bayer).squeeze().cpu().to(torch.float32) for m in methods.values()]
    recs = torch.stack(recs, 0)  # (B,C,H,W)

    # Original image
    orig = torch.tensor(orig).permute(2, 0, 1).to(torch.float32) / max_intensity
    orig = orig.unsqueeze(0).repeat(recs.shape[0], 1, 1, 1)

    # Compare
    psnr, eq_mask = debayer.metrics.peak_signal_noise_ratio(orig, recs, datarange=1.0)

    return psnr, eq_mask


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
    if "all" in args.methods:
        args.methods = ALL_METHODS
    methods = {
        "Debayer2x2": debayer.Debayer2x2(layout=LAYOUT).to(args.dev).to(prec),
        "Debayer3x3": debayer.Debayer3x3(layout=LAYOUT).to(args.dev).to(prec),
        "DebayerSplit": debayer.DebayerSplit(layout=LAYOUT).to(args.dev).to(prec),
        "Debayer5x5": debayer.Debayer5x5(layout=LAYOUT).to(args.dev).to(prec),
        "OpenCV": run_opencv,
    }
    methods = {k: v for k, v in methods.items() if k in args.methods}
    _logger.info(f"Enabled methods {list(methods.keys())}")

    results = []
    for path in image_paths:
        _logger.info(f"Processing {path}")
        psnrs, eqmasks = reconstruct(path, methods, args.dev, prec)
        for method, psnr, eqmask in zip(methods.keys(), psnrs, eqmasks):
            results.append(
                {
                    "Path": path,
                    "Database": args.path.name,
                    "Method": method,
                    "R (dB)": psnr[0].item(),
                    "G (dB)": psnr[1].item(),
                    "B (dB)": psnr[2].item(),
                    "PSNR (dB)": psnr.mean().item(),
                    "Equal": eqmask.any().item(),
                }
            )
    import pandas as pd

    df = pd.DataFrame(results)
    print()
    print(
        df.groupby([df.Database, df.Method])[
            ["Database", "Method", "R (dB)", "G (dB)", "B (dB)", "PSNR (dB)"]
        ]
        .mean()
        .reset_index()
        .to_markdown(headers="keys", index=False, floatfmt=".2f", tablefmt="github")
    )


if __name__ == "__main__":
    main()
