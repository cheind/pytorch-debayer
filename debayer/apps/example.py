import argparse
import cv2
import torch
import matplotlib.pyplot as plt

import debayer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["Debayer2x2", "Debayer3x3", "Debayer5x5", "DebayerSplit"],
        default="Debayer5x5",
        help="Debayer algorithm to use.",
    )
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("image")
    args = parser.parse_args()

    # Read Bayer image
    b = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    # Init filter
    deb = {
        "Debayer2x2": debayer.Debayer2x2,
        "Debayer3x3": debayer.Debayer3x3,
        "Debayer5x5": debayer.Debayer5x5,
        "DebayerSplit": debayer.DebayerSplit,
    }[args.method]()
    deb = deb.to(args.dev)

    # Prepare input with shape Bx1xHxW and
    t = (
        torch.from_numpy(b).to(torch.float32).unsqueeze(0).unsqueeze(0).to(args.dev)
    ) / 255.0

    # Compute and move back to CPU
    rgb = deb(t).squeeze().permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(
        cv2.cvtColor(b, cv2.COLOR_BAYER_BG2RGB) / 255.0, interpolation="nearest"
    )
    axs[0].set_title("OpenCV")
    axs[1].imshow(rgb, interpolation="nearest")
    axs[1].set_title(f"pytorch-debayer {args.method}")
    plt.show()


if __name__ == "__main__":
    main()
