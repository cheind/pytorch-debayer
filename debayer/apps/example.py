import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import debayer
from . import utils


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["Debayer2x2", "Debayer3x3", "Debayer5x5", "DebayerSplit"],
        default="Debayer5x5",
        help="Debayer algorithm to use.",
    )
    parser.add_argument("--half", action="store_true", help='Use 16bit fp precision')
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("image")
    args = parser.parse_args()    

    # Read Bayer image
    b: np.ndarray = plt.imread(args.image)
    if b.ndim > 2:
        b = b[..., 0]

    # Setup precision
    prec = torch.float16 if args.half else torch.float32

    # Init filter
    deb = {
        "Debayer2x2": debayer.Debayer2x2,
        "Debayer3x3": debayer.Debayer3x3,
        "Debayer5x5": debayer.Debayer5x5,
        "DebayerSplit": debayer.DebayerSplit,
    }[args.method]()
    deb = deb.to(args.dev).to(prec)

    # Prepare input with shape Bx1xHxW and
    t = (
        torch.tensor(b).to(prec).unsqueeze(0).unsqueeze(0).to(args.dev)
    ) / 255.0

    # Compute and move back to CPU
    rgb = deb(t).squeeze().permute(1, 2, 0).cpu().to(torch.float32).numpy()

    fig = utils.create_mosaic(
        [b, rgb], roi=(0, rgb.shape[1], 0, rgb.shape[0]), labels=["Bayer", args.method]
    )
    fig.savefig(f"tmp/{Path(args.image).with_suffix('.png').name}")
    plt.show()


if __name__ == "__main__":
    main()
