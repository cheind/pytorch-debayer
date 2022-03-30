# Additional Evaluation Results

Metrics are evaluated on different full-color datasets such as
 - Kodak  
 http://r0k.us/graphics/kodak/
 https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite
 - McMaster
 https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm

To run the evaluation, make each dataset a folder and run
```
python -m debayer.apps.eval <Database-Path> --methods all
```

Note, in case the latest version below does not match the current version of pytorch-debayer, performance statistics to remain unchanged.

## Version 1.4.0

| Database   | Method       |     R |     G |     B |   PSNR |
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