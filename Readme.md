## pytorch-debayer

Provides GPU demosaicing of images captured with Bayer color filter arrays (CFA) with batch support. This implementation relies on pure PyTorch functionality and thus avoids any extra build steps.

Currently, the following methods are provided
 - `debayer.Debayer2x2` uses 2x2 convolutions. Trades speed for color accuracy.
 - `debayer.Debayer3x3` uses 3x3 convolutions. Slower but reconstruction results comparable with `OpenCV.cvtColor`.
 - `debayer.Debayer5x5` uses 5x5 convolutions based on Malver-He-Cutler algorithm. Slower but sharper than `OpenCV.cvtColor`.
 - `debayer.DebayerSplit` faster than Debayer3x3 but decreased image quality.

As of `v1.0.2` the method `debayer.Debayer5x5` should be used per default.

This library is most useful when downstream image processing happens with PyTorch models. Additionally the upload of Bayer images reduces the occupied bandwidth compared to color images.

### Usage
Usage is straight forward

```python
import torch
from debayer import Debayer5x5

f = Debayer5x5().cuda()

bayer = ...         # a Bx1xHxW, torch.float32 tensor of BG-Bayer images
with torch.no_grad():
    rgb = f(bayer)  # a Bx3xHxW, torch.float32 tensor of RGB images
```

see [this example](debayer/apps/example.py) for elaborate code.

### Install
```
pip install git+https://github.com/cheind/pytorch-debayer
```

### Limitations

Currently **pytorch-debayer** requires
 - BG-Bayer color filter array layout. According to OpenCV naming conventions (see [here](https://docs.opencv.org/4.2.0/de/d25/imgproc_color_conversions.html) towards end of file) that means your Bayer input image must be arranged in the following way
 ```
RGRGRG...
GBGBGB...
RGRGRG...
.........
```
 - the number of image rows and columns to be even.

### Benchmark
Performance comparison using a 5 megapixel [test image](etc/test.bmp).

Method | Device | Elapsed [msec/image] | Mode |
|:----:|:------:|:-------:|:----|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.71 | time_upload=False,batch_size=10 |
| Debayer2x2 | GeForce RTX 2080 SUPER | 0.52 | time_upload=False,batch_size=10 |
| Debayer2x2 | Tesla V100-SXM2-16GB | 0.31 | time_upload=False,batch_size=10 |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.80 | time_upload=False,batch_size=10 |
| Debayer3x3 | GeForce RTX 2080 SUPER | 2.16 | time_upload=False,batch_size=10 |
| Debayer3x3 | Tesla V100-SXM2-16GB | 1.21 | time_upload=False,batch_size=10 |
| DebayerSplit | GeForce GTX 1080 Ti | 2.54 | time_upload=False,batch_size=10 |
| DebayerSplit | Tesla V100-SXM2-16GB | 1.08 | time_upload=False,batch_size=10 |
| OpenCV 4.1.2 | CPU i7-8700K | 3.13 | transparent_api=False,time_upload=False,batch_size=10 |
| OpenCV 4.1.2 | GPU GeForce GTX 1080 Ti | 1.25 | transparent_api=True,time_upload=False,batch_size=10 |

Stats computed by [benchmark code](debayer/apps/benchmark.py). Invoke with

```
> python -m debayer.apps.benchmark etc\test.bmp
```

### Comparisons

Here are some subjective image demosaicing results using the following [test image](etc/test.bmp) image. 
<div align="center">
<img width="60%" src="etc/readme/input.png" />
</div>

The following highlights algorithmic differences on various smaller regions for improved pixel visibility. From left to right 
```
OpenCV, Debayer2x2, Debayer3x3, DebayerSplit, Debayer5x5
```

Click images to enlarge.

<div align="center">
<img width="100%" src="etc/readme/test-mosaic-l1429-r1659-b1889-t1725.png" />
</div>

<div align="center">
<img width="100%" src="etc/readme/test-mosaic-l1779-r1998-b1145-t949.png" />
</div>

<div align="center">
<img width="100%" src="etc/readme/test-mosaic-l620-r872-b1430-t1233.png" />
</div>

<div align="center">
<img width="100%" src="etc/readme/test-mosaic-l588-r817-b1178-t981.png" />
</div>

Created using
```
python -m debayer.apps.compare etc\test.bmp
# Then select a region and check `tmp`/
```