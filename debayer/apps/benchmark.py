import logging
import argparse
import time

import cpuinfo
import torch
from itertools import product

from . import utils

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    print("Skipping OpenCV, not installed.")
    OPENCV_AVAILABLE = False

import debayer


def run_pytorch(deb, t, dev, kwargs: dict):
    time_upload = kwargs.get("time_upload", False)
    B = kwargs.get("batch", 10)
    runs = kwargs.get("runs", 100)
    prec = kwargs.get("prec", torch.float32)

    t = t.repeat(B, 1, 1, 1).to(prec).contiguous()
    t = t.pin_memory() if time_upload else t.to(dev)

    def run_once():
        x = t.to(dev, non_blocking=True) if time_upload else t
        rgb = deb(x)  # noqa

    # Warmup
    run_once()
    run_once()
    run_once()

    if dev != "cpu":
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
    else:
        t0 = time.perf_counter()
        N = runs
        for _ in range(N):
            run_once()
        return (time.perf_counter() - t0) / (runs * B) * 1e3


def run_opencv(b, **kwargs):
    # see https://www.learnopencv.com/opencv-transparent-api/
    num_threads = kwargs.get("opencv-threads", None)
    if num_threads is not None:
        cv2.setNumThreads(num_threads)
    time_upload = kwargs.get("time_upload", False)
    transparent_api = kwargs.get("transparent_api", False)
    runs = kwargs.get("runs", 100)
    layout = kwargs.get("layout", debayer.Layout.RGGB)
    B = kwargs.get("batch_size", 10)

    b = cv2.UMat(b) if (transparent_api and not time_upload) else b

    def run_once():
        x = cv2.UMat(b) if (transparent_api and time_upload) else b
        y = cv2.cvtColor(x, utils.opencv_conversion_code(layout))
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
        z = y.get()  # noqa
    return (time.perf_counter() - start) / N * 1e3


def fmt_line(method, devname, elapsed, **modeargs):
    modeargs.pop("batch_size", None)
    modeargs.pop("runs", None)
    modeargs.pop("layout", None)
    modeargs.pop("bayer", None)
    modeargs.pop("image", None)
    modeargs.pop("methods", None)
    modeargs.pop("dev", None)
    mode = ",".join([f"{k}={v}" for k, v in modeargs.items()])
    return f"| {method} | {devname} | {elapsed:5.3f} | {mode} |"


@torch.no_grad()
def bench_debayer(b, args):
    if args.dev != "cpu":
        devname = torch.cuda.get_device_name(args.dev)
    else:
        devname = cpuinfo.get_cpu_info()["brand_raw"]
    mode = vars(args)

    def run_all(dev, mode):
        t = (torch.tensor(b).clone().unsqueeze(0).unsqueeze(0)) / 255.0
        prec = mode["prec"]
        mods = {
            "Debayer2x2": debayer.Debayer2x2(layout=mode["layout"]).to(prec).to(dev),
            "Debayer3x3": debayer.Debayer3x3(layout=mode["layout"]).to(prec).to(dev),
            "Debayer5x5": debayer.Debayer5x5(layout=mode["layout"]).to(prec).to(dev),
            "DebayerSplit": debayer.DebayerSplit(layout=mode["layout"])
            .to(prec)
            .to(dev),
        }
        mods = {k: v for k, v in mods.items() if k in args.methods}
        if mode["torchscript"]:
            mods = {
                k: torch.jit.script(
                    torch.jit.trace(v, torch.rand(1, 1, 128, 128).to(dev).to(prec))
                )
                for k, v in mods.items()
            }

        for name, mod in mods.items():
            e = run_pytorch(mod, t, dev, mode)
            print(fmt_line(name, devname, e, **mode))

    for prec, script in product([torch.float32, torch.float16], [True, False]):
        run_all(args.dev, mode={**mode, **{"prec": prec, "torchscript": script}})


def bench_opencv(b, args):
    if "OpenCV" not in args.methods:
        return
    devname = cpuinfo.get_cpu_info()["brand_raw"]

    for threads, transparent in product([4, 12], [False, True]):
        mode = {**vars(args), "opencv-threads": threads, "transparent-api": transparent}
        e = run_opencv(b, **mode)
        print(fmt_line(f"OpenCV {cv2.__version__}", devname, e, **mode))


ALL_METHODS = ["Debayer2x2", "Debayer3x3", "Debayer5x5", "DebayerSplit", "OpenCV"]


def main():
    logging.basicConfig(level=logging.INFO)
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
    if "all" in args.methods:
        args.methods = ALL_METHODS

    input_image, b = utils.read_image(args.image, bayer=args.bayer, layout=args.layout)

    print(f"torch: v{torch.__version__}")
    print(f"pytorch-debayer: v{debayer.__version__}")

    print()
    mpix = b.size * 1e-6
    print(f"Method | Device | Elapsed [msec / {mpix:.1f}mpix] | Mode |")
    print("|:----:|:------:|:-------:|:----:|")

    bench_debayer(b, args)
    if OPENCV_AVAILABLE:
        bench_opencv(b, args)


if __name__ == "__main__":
    main()
