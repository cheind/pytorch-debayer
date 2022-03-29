import argparse
import logging
from pathlib import Path

import cv2
import debayer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch

from . import utils


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("--half", action="store_true", help="Use 16bit fp precision")
    parser.add_argument("--save-zoom", action="store_true", help="Save zoom regions")
    parser.add_argument(
        "--layout",
        type=debayer.Layout,
        choices=list(debayer.Layout),
        default=debayer.Layout.RGGB,
        help="Bayer layout of Bayer input image. Only applicable if --full-color is omitted",  # noqa
    )
    parser.add_argument(
        "--bayer",
        action="store_true",
        help="If input image is multi-channel, assume encoding is Bayer",
    )
    parser.add_argument("image")
    args = parser.parse_args()

    prec = torch.float16 if args.half else torch.float32

    methods = {
        "Debayer2x2": debayer.Debayer2x2(layout=args.layout).to(args.dev).to(prec),
        "Debayer3x3": debayer.Debayer3x3(layout=args.layout).to(args.dev).to(prec),
        "DebayerSplit": debayer.DebayerSplit(layout=args.layout).to(args.dev).to(prec),
        "Debayer5x5": debayer.Debayer5x5(layout=args.layout).to(args.dev).to(prec),
    }

    # Read Bayer image
    input_image, b = utils.read_image(args.image, bayer=args.bayer, layout=args.layout)

    # Compute debayer results
    # Prepare input with shape Bx1xHxW and
    t = (torch.from_numpy(b).to(prec).unsqueeze(0).unsqueeze(0).to(args.dev)) / 255.0
    res = {
        **{
            "Original": input_image,
            "OpenCV": cv2.cvtColor(b, utils.opencv_conversion_code(args.layout))
            / 255.0,
        },
        **{
            k: deb(t).squeeze().permute(1, 2, 0).to(torch.float32).cpu().numpy()
            for k, deb in methods.items()
        },
    }

    nrows = 2
    ncols = 3
    h, w = b.shape
    W, H = plt.figaspect((h * nrows) / (w * ncols))

    fig = plt.figure(constrained_layout=True, figsize=(W, H))
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    spec.update(wspace=0, hspace=0)

    ax = None
    for idx, (key, img) in enumerate(res.items()):
        ax = fig.add_subplot(spec[idx], sharey=ax, sharex=ax)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(key, size=10, y=1.0, pad=-14, color="white")
        ax.set_frame_on(False)
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )

    def create_image_mosaic(event_ax):
        left, right = event_ax.get_xlim()
        top, bottom = event_ax.get_ylim()
        return utils.create_mosaic(
            list(res.values()), (left, right, bottom, top), list(res.keys())
        )

    def on_ylims_change(event_ax):
        fig = create_image_mosaic(event_ax)

        left, right = event_ax.get_xlim()
        bottom, top = event_ax.get_ylim()

        left, right, top, bottom = map(int, [left, right, top, bottom])

        iname = Path(args.image).with_suffix("").name
        path = f"tmp/{iname}-mosaic-l{left}-r{right}-b{bottom}-t{top}.png"
        print("Saved to", path)
        fig.savefig(path)
        plt.close(fig)

    if args.save_zoom:
        ax.callbacks.connect("ylim_changed", on_ylims_change)

    plt.show()


if __name__ == "__main__":
    main()
