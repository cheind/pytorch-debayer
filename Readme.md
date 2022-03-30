# pytorch-debayer

Provides GPU demosaicing of images captured with Bayer color filter arrays (CFA) with batch support. This implementation relies on pure PyTorch functionality and thus avoids any extra build steps. This library is most useful when downstream image processing happens with PyTorch models. Additionally, uploading of Bayer images (instead of RGB) significantly reduces the occupied bandwidth.

## Features
 - **Methods** Currently, the following methods are provided
    - `debayer.Debayer2x2` uses 2x2 convolutions. Trades speed for color accuracy.
    - `debayer.Debayer3x3` uses 3x3 convolutions. Slower but reconstruction results comparable with `OpenCV.cvtColor`.
    - `debayer.Debayer5x5` uses 5x5 convolutions based on Malver-He-Cutler algorithm. Slower but sharper than `OpenCV.cvtColor`. Should be your default.
    - `debayer.DebayerSplit` faster than Debayer3x3 but decreased image quality.    
 - **Precision** Each method supports `float32` or `float16` precision. 

## Usage
Usage is straight forward

```python
import torch
from debayer import Debayer5x5

f = Debayer5x5().cuda()

bayer = ...         # a Bx1xHxW, [0..1], torch.float32 RGGB-Bayer tensor
with torch.no_grad():
    rgb = f(bayer)  # a Bx3xHxW, torch.float32 tensor of RGB images
```

see [this example](debayer/apps/example.py) for elaborate code.

## Install
Library, apps and development tools
```
pip install git+https://github.com/cheind/pytorch-debayer#egg=pytorch-debayer[full]
```

Just the library core requirements
```
pip install git+https://github.com/cheind/pytorch-debayer
```

## Bayer Layouts
Bayer filter arrays may come in different layouts. **pytorch-debayer** distinguishes these layouts by looking at the upper-left 2x2 pixel block. For example
```
RGrg...
GBgb...
rgrg...
```
defines the `Layout.RGGB` which is also the default. In total four layouts are supported
```python
from debayer import Layout

Layout.RGGB
Layout.GRBG
Layout.GBRG
Layout.BGGR
```

and you can set the layout as follows

```python
from debayer import Debayer5x5, Layout

f = Debayer5x5(layout=Layout.BGGR).cuda()
```

## Evaluation

### PSNR values
The PSNR (Peak-Signal-Noise-Ratio) values (dB, higher is better) for each channel (R, G, B) and PSNR of the whole image (RGB) across 2 Datasets (Kodak, McMaster) and for each algorithm. See [Metrics.md](./Metrics.md) for additional details.

| Database   | Method       |     R (dB)|     G (dB)|     B (dB)|   PSNR (dB)|
|------------|--------------|-------|-------|-------|--------|
| Kodak      | Debayer2x2   | 26.64 | 28.18 | 26.98 |  27.27 |
|       | Debayer3x3   | 28.18 | 32.66 | 28.86 |  29.90 |
|       | Debayer5x5   | 33.84 | 38.05 | 33.53 |  35.14 |
|       | DebayerSplit | 26.64 | 32.66 | 26.98 |  28.76 |
|       | OpenCV       | 28.15 | 31.25 | 28.62 |  29.34 |
| McMaster   | Debayer2x2   | 28.47 | 30.32 | 28.63 |  29.14 |
|    | Debayer3x3   | 31.68 | 35.40 | 31.25 |  32.78 |
|    | Debayer5x5   | 34.04 | 37.62 | 33.02 |  34.89 |
|    | DebayerSplit | 28.47 | 35.40 | 28.63 |  30.83 |
|    | OpenCV       | 31.64 | 35.22 | 31.22 |  32.69 |


### Runtimes
Performance comparison on a 5 megapixel [test image](etc/test.bmp) using a batch size of 10. 
Timings are in milliseconds per given megapixels. See [Benchmarks.md](./Benchmarks.md) for additional details.

Method | Device | Elapsed [msec/5.1mpix] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.653 | prec=torch.float32,time_upload=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.056 | prec=torch.float32,time_upload=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 6.343 | prec=torch.float32,time_upload=False |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.562 | prec=torch.float16,time_upload=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.812 | prec=torch.float16,time_upload=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.545 | prec=torch.float16,time_upload=False |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.232 | prec=torch.float32,time_upload=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.173 | prec=torch.float32,time_upload=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.723 | prec=torch.float32,time_upload=False |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.173 | prec=torch.float16,time_upload=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.067 | prec=torch.float16,time_upload=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.687 | prec=torch.float16,time_upload=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.097 | transparent_api=False,time_upload=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 11.042 | transparent_api=True,time_upload=False |


## Comparisons

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

## Limitations

Currently **pytorch-debayer** requires
 - the image to have an even number of rows and columns
 - `debayer.DebayerSplit` requires a Bayer filter layout of `Layout.RGGB`, all others support varying layouts (since v1.3.0).

## References

Losson, Olivier, Ludovic Macaire, and Yanqin Yang. "Comparison of color demosaicing methods." Advances in Imaging and electron Physics. Vol. 162. Elsevier, 2010. 173-265.