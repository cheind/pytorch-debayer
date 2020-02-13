## pytorch-debayer

Provides GPU demosaicing of images captured with Bayer color filter arrays (CFA) with batch support. This implementation relies on pure PyTorch functionality and thus avoids any extra build steps.

Currently, two modules based on bilinear interpolation are provided
 - `debayer.Debayer2x2` uses 2x2 convolutions. Trades speed for color accuracy.
 - `debayer.Debayer3x3` uses 3x3 convolutions. Slower but reconstruction results comparable with `OpenCV.cvtColor`.

This library is most useful when downstream image processing happens with PyTorch models. Additionally the upload of Bayer images reduces the occupied bandwidth compared to color images.

### Usage
Usage is straight forward

```python
import torch
from debayer import Debayer3x3

f = Debayer3x3().cuda()

bayer = ...     # a Bx1xHxW, torch.float32 tensor of BG-Bayer images
rgb = f(bayer)  # a Bx3xHxW, torch.float32 tensor of RGB images
```

see [this example](debayer/apps/example.py) for elaborate code.

### Install
```
pip install git+https://github.com/cheind/pytorch-debayer
```

### Limitations

Currently **pytorch-debayer** requires BG-Bayer color filter array layout. According to OpenCV naming conventions (see [here](https://docs.opencv.org/4.2.0/de/d25/imgproc_color_conversions.html) towards end of file) that means your Bayer input image must be arranged in the following way
```
RGRGRG...
GBGBGB...
RGRGRG...
```

### Benchmark
Performance comparison using a 5 megapixel [test image](etc/test.bmp).

Method | Device | Elapsed | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.71 msec/image | time_upload=False,batch_size=10 |
| Debayer2x2 | GeForce RTX 2080 SUPER | 0.52 msec/image | time_upload=False,batch_size=10 |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.80 msec/image | time_upload=False,batch_size=10 |
| Debayer3x3 | GeForce RTX 2080 SUPER | 2.16 msec/image | time_upload=False,batch_size=10 |
| OpenCV 4.1.2 | CPU i7-8700K | 3.13 msec/image | transparent_api=False,time_upload=False,batch_size=10 |
| OpenCV 4.1.2 | GPU GeForce GTX 1080 Ti | 1.25 msec/image | transparent_api=True,time_upload=False,batch_size=10 |

Stats computed by [benchmark code](debayer/apps/benchmark.py). Invoke with

```
> python -m debayer.apps.benchmark etc\test.bmp
```
