import torch
import numpy as np
import argparse
import time
from PIL import Image

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    print("Skipping OpenCV, not installed.")
    OPENCV_AVAILABLE = False

import debayer


def run_pytorch(deb, t, dev, prec, **kwargs):
    time_upload = kwargs.get("time_upload", False)
    B = kwargs.get("batch_size", 10)
    runs = kwargs.get("runs", 100)

    t = t.repeat(B, 1, 1, 1).to(prec)
    t = t.pin_memory() if time_upload else t.to(dev)

    def run_once():
        x = t.to(dev, non_blocking=True) if time_upload else t
        rgb = deb(x)

    # Warmup
    run_once()
    run_once()
    run_once()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    N = runs
    for _ in range(N):
        run_once()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / (runs * B)


def run_opencv(b, **kwargs):
    # see https://www.learnopencv.com/opencv-transparent-api/
    time_upload = kwargs.get("time_upload", False)
    transparent_api = kwargs.get("transparent_api", False)
    runs = kwargs.get("runs", 100)
    B = kwargs.get("batch_size", 10)

    b = cv2.UMat(b) if (transparent_api and not time_upload) else b

    def run_once():
        x = cv2.UMat(b) if (transparent_api and time_upload) else b
        y = cv2.cvtColor(x, cv2.COLOR_BAYER_BG2RGB)
        return y

    run_once()
    run_once()
    y = run_once()
    if transparent_api:
        z = y.get()

    N = runs * B
    start = time.perf_counter()
    for _ in range(N):
        y = run_once()
    if transparent_api:
        z = y.get()
    return (time.perf_counter() - start) / N * 1e3


def fmt_line(method, devname, elapsed, **modeargs):
    modeargs.pop("batch_size", None)
    modeargs.pop("runs", None)
    mode = ",".join([f"{k}={v}" for k, v in modeargs.items()])
    return f"| {method} | {devname} | {elapsed:5.3f} | {mode} |"


@torch.no_grad()
def bench_debayer(b, args):
    devname = torch.cuda.get_device_name(args.dev)
    mode = dict(time_upload=args.time_upload, batch_size=args.batch, runs=args.runs)

    def run_all(dev, mode):
        t = (torch.tensor(b).clone().unsqueeze(0).unsqueeze(0)) / 255.0
        prec = mode["prec"]

        if "Debayer2x2" in args.methods:
            deb = debayer.Debayer2x2().to(dev).to(prec)
            debname = deb.__class__.__name__
            e = run_pytorch(deb, t, dev, **mode)
            print(fmt_line(debname, devname, e, **mode))

        if "Debayer3x3" in args.methods:
            deb = debayer.Debayer3x3().to(dev).to(prec)
            debname = deb.__class__.__name__
            e = run_pytorch(deb, t, dev, **mode)
            print(fmt_line(debname, devname, e, **mode))

        if "Debayer5x5" in args.methods:
            deb = debayer.Debayer5x5().to(dev).to(prec)
            debname = deb.__class__.__name__
            e = run_pytorch(deb, t, dev, **mode)
            print(fmt_line(debname, devname, e, **mode))

        if "DebayerSplit" in args.methods:
            deb = debayer.DebayerSplit().to(dev).to(prec)
            debname = deb.__class__.__name__
            e = run_pytorch(deb, t, dev, **mode)
            print(fmt_line(debname, devname, e, **mode))

    run_all(args.dev, mode={**{"prec": torch.float32}, **mode})
    run_all(args.dev, mode={**{"prec": torch.float16}, **mode})


def bench_opencv(b, args):
    if not "OpenCV" in args.methods:
        return
    devname = torch.cuda.get_device_name(args.dev)
    mode = dict(
        transparent_api=False,
        time_upload=args.time_upload,
        batch_size=args.batch,
        runs=args.runs,
    )
    e = run_opencv(b, **mode)
    print(fmt_line(f"OpenCV {cv2.__version__}", "CPU/OpenCL", e, **mode))
    mode = dict(
        transparent_api=True,
        time_upload=args.time_upload,
        batch_size=args.batch,
        runs=args.runs,
    )
    e = run_opencv(b, **mode)
    print(fmt_line(f"OpenCV {cv2.__version__}", "CPU/OpenCL", e, **mode))


ALL_METHODS = ["Debayer2x2", "Debayer3x3", "Debayer5x5", "DebayerSplit", "OpenCV"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda")
    parser.add_argument("--batch", default=10, type=int)
    parser.add_argument("--time-upload", action="store_true")
    parser.add_argument("--runs", type=int, default=100, help="Number runs")
    parser.add_argument(
        "--methods",
        default=["Debayer3x3", "Debayer5x5", "OpenCV"],
        nargs="+",
        help="Which methods to run. List of methods or `all`.",
        choices=ALL_METHODS + ["all"],
    )
    parser.add_argument("image")
    args = parser.parse_args()
    if args.methods == "all":
        args.methods = ALL_METHODS

    b = np.asarray(Image.open(args.image).convert("L"))

    print(f"running pytorch-debayer: {debayer.__version__}")
    print()
    print("Method | Device | Elapsed [msec/image] | Mode |")
    print("|:----:|:------:|:-------:|:----:|")

    bench_debayer(b, args)
    if OPENCV_AVAILABLE:
        bench_opencv(b, args)


if __name__ == "__main__":
    main()
