import argparse
from configparser import Interpolation
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import debayer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("image")
    args = parser.parse_args()

    methods = {
        "Debayer2x2": debayer.Debayer2x2().to(args.dev),
        "Debayer3x3": debayer.Debayer3x3().to(args.dev),
        "Debayer5x5": debayer.Debayer5x5().to(args.dev),
        "DebayerSplit": debayer.DebayerSplit().to(args.dev),
    }

    # Read Bayer image
    b = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    # Compute OpenCV result
    rgb_opencv = cv2.cvtColor(b, cv2.COLOR_BAYER_BG2RGB) / 255.0

    # Compute debayer results
    # Prepare input with shape Bx1xHxW and
    t = (
        torch.from_numpy(b).to(torch.float32).unsqueeze(0).unsqueeze(0).to(args.dev)
    ) / 255.0

    res = {
        k: deb(t).squeeze().permute(1, 2, 0).cpu().numpy() for k, deb in methods.items()
    }

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    def show_image(fig, spec, row, col, img, title, share_ax=None):
        ax = fig.add_subplot(spec[row, col], sharey=share_ax, sharex=share_ax)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title)
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        return ax

    ax0 = show_image(fig, spec, 0, 0, rgb_opencv, "OpenCV")
    show_image(fig, spec, 0, 1, res["Debayer2x2"], "Debayer2x2", ax0)
    show_image(fig, spec, 0, 2, res["DebayerSplit"], "DebayerSplit", ax0)
    show_image(fig, spec, 1, 0, res["Debayer3x3"], "Debayer3x3", ax0)
    show_image(fig, spec, 1, 1, res["Debayer5x5"], "Debayer5x5", ax0)
    fig.suptitle("Comparison of Demosaicing Methods", size=20)
    spec.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
