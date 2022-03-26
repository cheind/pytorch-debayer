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

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    def show_image(fig, spec, row, col, img, title, share_ax=None):
        ax = fig.add_subplot(spec[row, col], sharey=share_ax, sharex=share_ax)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, size=8)
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        return ax

    #
    axcv = show_image(fig, spec, 0, 1, rgb_opencv, "OpenCV")
    show_image(fig, spec, 0, 2, res["Debayer2x2"], "Debayer2x2", axcv)
    show_image(fig, spec, 1, 0, res["DebayerSplit"], "DebayerSplit", axcv)
    show_image(fig, spec, 1, 1, res["Debayer3x3"], "Debayer3x3", axcv)
    show_image(fig, spec, 1, 2, res["Debayer5x5"], "Debayer5x5", axcv)

    fig.suptitle("Comparison of Demosaicing Methods", size=10)
    spec.tight_layout(fig, rect=[0, 0, 1, 0.97])

    # axzoom = show_image(fig, spec, 0, 0, res["Debayer5x5"], "Zoom")
    # def update_zoom(event_ax):
    #     axzoom.clear()
    #     axzoom.imshow(
    #         rgb_opencv,
    #         extent=(0, rgb_opencv.shape[1], rgb_opencv.shape[0], 0),
    #         interpolation="nearest",
    #         origin="upper",
    #     )

    #     left, right = event_ax.get_xlim()
    #     bottom, top = event_ax.get_ylim()
    #     axins = axzoom.inset_axes([0.5, 0.5, 0.47, 0.47])
    #     axins.imshow(
    #         rgb_opencv,
    #         extent=(0, rgb_opencv.shape[1], rgb_opencv.shape[0], 0),
    #         origin="upper",
    #     )
    #     axins.tick_params(
    #         bottom=False,
    #         labelbottom=False,
    #         left=False,
    #         labelleft=False,
    #     )
    #     axins.set_xlim(left, right)
    #     axins.set_ylim(bottom, top)
    #     axzoom.indicate_inset_zoom(axins, edgecolor="red")
    #     axzoom.tick_params(
    #         bottom=False,
    #         top=False,
    #         left=False,
    #         right=False,
    #         labelleft=False,
    #         labelbottom=False,
    #     )
    # def on_ylims_change(event_ax):
    #     update_zoom(event_ax)
    # axcv.callbacks.connect("ylim_changed", on_ylims_change)

    plt.show()


if __name__ == "__main__":
    main()
