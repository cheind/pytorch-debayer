import torch
import numpy as np
import argparse
import time
from PIL import Image

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print('Skipping OpenCV, not installed.')
    OPENCV_AVAILABLE = False

import debayer

def run_pytorch(deb, t, dev, **kwargs):
    time_upload = kwargs.get('time_upload', False)
    B = kwargs.get('batch_size', 10)

    t = t.repeat(B,1,1,1)
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
    N = 1000
    for _ in range(N):
        run_once()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)/(N*B)

def run_opencv(b, **kwargs):
    # see https://www.learnopencv.com/opencv-transparent-api/
    time_upload = kwargs.get('time_upload', False)
    transparent_api = kwargs.get('transparent_api', False)
    B = kwargs.get('batch_size', 10)
    
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

    N = 1000*B
    start = time.time()
    for _ in range(N):
        y = run_once()        
    if transparent_api:
        z = y.get()
    return (time.time() - start)/N*1000

def fmt_line(method, devname, elapsed, **modeargs):
    mode = ','.join([f'{k}={v}' for k,v in modeargs.items()])
    return f'| {method} | {devname} | {elapsed:4.2f} | {mode} |'

@torch.no_grad()
def bench_debayer(b, args):
    devname = torch.cuda.get_device_name(args.dev)
    mode = dict(time_upload=args.time_upload, batch_size=args.batch)

    t = (
        torch.from_numpy(b)
        .unsqueeze(0)
        .unsqueeze(0)        
    ) / 255.0

    deb = debayer.Debayer2x2().to(args.dev)
    deb = deb.to(args.dev)
    debname = deb.__class__.__name__        
    e = run_pytorch(deb, t, args.dev, **mode)
    print(fmt_line(debname, devname, e, **mode))
    
    deb = debayer.Debayer3x3().to(args.dev)
    deb = deb.to(args.dev)
    debname = deb.__class__.__name__
    e = run_pytorch(deb, t, args.dev, **mode)
    print(fmt_line(debname, devname, e, **mode))

    deb = debayer.DebayerSplit().to(args.dev)
    deb = deb.to(args.dev)
    debname = deb.__class__.__name__
    e = run_pytorch(deb, t, args.dev, **mode)
    print(fmt_line(debname, devname, e, **mode))

def bench_opencv(b, args):
    mode = dict(transparent_api=False, time_upload=args.time_upload, batch_size=args.batch)
    e = run_opencv(b, **mode)
    print(fmt_line(f'OpenCV {cv2.__version__}', 'CPU ??', e, **mode))
    mode = dict(transparent_api=True, time_upload=args.time_upload, batch_size=args.batch)
    e = run_opencv(b, **mode)
    print(fmt_line(f'OpenCV {cv2.__version__}', 'GPU ??', e, **mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', default='cuda')
    parser.add_argument('--batch', default=10, type=int)
    parser.add_argument('--time-upload', action='store_true')
    parser.add_argument('image')
    args = parser.parse_args()

    b = np.asarray(Image.open(args.image).convert('L'))
    

    print(f'running pytorch-debayer: {debayer.__version__}')
    print()
    print('Method | Device | Elapsed [msec/image] | Mode |')
    print('|:----:|:------:|:-------:|:----:|')

    bench_debayer(b, args)
    if OPENCV_AVAILABLE:
        bench_opencv(b, args)

if __name__ == '__main__':
    main()
