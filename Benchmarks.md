# Additional Benchmark Results

Performance comparison on a 5 megapixel [test image](etc/test.bmp). Command is
```
python -m debayer.apps.benchmark etc/test.bmp --methods all
```
Note, in case the latest version below does not match the current version of pytorch-debayer, performance statistics to remain unchanged.

## Version 1.4.1
torch: v1.10.0+cu113
pytorch-debayer: v1.4.1

Method | Device | Elapsed [msec / 5.1mpix] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.619 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.296 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer5x5 | GeForce GTX 1080 Ti | 5.837 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| DebayerSplit | GeForce GTX 1080 Ti | 2.631 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.617 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.298 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 5.842 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| DebayerSplit | GeForce GTX 1080 Ti | 2.632 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.561 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.919 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.045 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| DebayerSplit | GeForce GTX 1080 Ti | 1.151 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.563 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.927 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.044 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| DebayerSplit | GeForce GTX 1080 Ti | 1.151 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.205 | batch=10,time_upload=False,opencv-threads=4,transparent-api=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.206 | batch=10,time_upload=False,opencv-threads=4,transparent-api=True |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 1.937 | batch=10,time_upload=False,opencv-threads=12,transparent-api=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 1.925 | batch=10,time_upload=False,opencv-threads=12,transparent-api=True |


## Version 1.3.0

### Machine 1
torch: v1.10.0+cu113
pytorch-debayer: v1.3.0

Method | Device | Elapsed [msec / 5.1mpix] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.633 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer3x3 | GeForce GTX 1080 Ti | 4.043 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer5x5 | GeForce GTX 1080 Ti | 6.719 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| DebayerSplit | GeForce GTX 1080 Ti | 2.689 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.631 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.874 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 6.487 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| DebayerSplit | GeForce GTX 1080 Ti | 2.697 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.575 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.498 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.640 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| DebayerSplit | GeForce GTX 1080 Ti | 1.173 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.574 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.498 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.640 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| DebayerSplit | GeForce GTX 1080 Ti | 1.173 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.221 | batch=10,time_upload=False,opencv-threads=4,transparent-api=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.202 | batch=10,time_upload=False,opencv-threads=4,transparent-api=True |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 1.968 | batch=10,time_upload=False,opencv-threads=12,transparent-api=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 1.956 | batch=10,time_upload=False,opencv-threads=12,transparent-api=True |

### Machine 2
torch: v1.11.0+cu113
pytorch-debayer: v1.3.0

Method | Device | Elapsed [msec / 5.1mpix] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.231 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.173 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.735 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 12.023 | batch=10,time_upload=False,prec=torch.float32,torchscript=True |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.231 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.173 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.733 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 11.966 | batch=10,time_upload=False,prec=torch.float32,torchscript=False |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.173 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 0.981 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.700 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 0.373 | batch=10,time_upload=False,prec=torch.float16,torchscript=True |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.175 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 0.982 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.701 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 0.373 | batch=10,time_upload=False,prec=torch.float16,torchscript=False |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 0.708 | batch=10,time_upload=False,opencv-threads=4,transparent-api=False |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 0.751 | batch=10,time_upload=False,opencv-threads=4,transparent-api=True |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 0.816 | batch=10,time_upload=False,opencv-threads=12,transparent-api=False |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 0.728 | batch=10,time_upload=False,opencv-threads=12,transparent-api=True |


## Version 1.1.0

### Machine 1
torch: v1.10.0+cu113
pytorch-debayer: v1.1.0

Method | Device | Elapsed [msec/image] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | GeForce GTX 1080 Ti | 0.653 | prec=torch.float32,time_upload=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 3.056 | prec=torch.float32,time_upload=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 6.343 | prec=torch.float32,time_upload=False |
| DebayerSplit | GeForce GTX 1080 Ti | 2.635 | prec=torch.float32,time_upload=False |
| Debayer2x2 | GeForce GTX 1080 Ti | 0.562 | prec=torch.float16,time_upload=False |
| Debayer3x3 | GeForce GTX 1080 Ti | 2.812 | prec=torch.float16,time_upload=False |
| Debayer5x5 | GeForce GTX 1080 Ti | 4.545 | prec=torch.float16,time_upload=False |
| DebayerSplit | GeForce GTX 1080 Ti | 1.148 | prec=torch.float16,time_upload=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 2.097 | transparent_api=False,time_upload=False |
| OpenCV 4.5.3 | Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz | 11.042 | transparent_api=True,time_upload=False |

### Machine 2
torch: v1.11.0+cu113
pytorch-debayer: v1.1.0

Method | Device | Elapsed [msec/image] | Mode |
|:----:|:------:|:-------:|:----:|
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.232 | prec=torch.float32,time_upload=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.173 | prec=torch.float32,time_upload=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.723 | prec=torch.float32,time_upload=False |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 11.959 | prec=torch.float32,time_upload=False |
| Debayer2x2 | NVIDIA GeForce RTX 3090 | 0.173 | prec=torch.float16,time_upload=False |
| Debayer3x3 | NVIDIA GeForce RTX 3090 | 1.067 | prec=torch.float16,time_upload=False |
| Debayer5x5 | NVIDIA GeForce RTX 3090 | 1.687 | prec=torch.float16,time_upload=False |
| DebayerSplit | NVIDIA GeForce RTX 3090 | 0.371 | prec=torch.float16,time_upload=False |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 0.696 | transparent_api=False,time_upload=False |
| OpenCV 4.5.5 | Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz | 5.216 | transparent_api=True,time_upload=False |


## Version 1.0.0

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