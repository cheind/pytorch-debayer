# Additional Benchmark Results

Performance comparison on a 5 megapixel [test image](etc/test.bmp). Command is
```
python -m debayer.apps.benchmark etc\test.bmp --methods all
```

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