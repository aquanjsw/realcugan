# RealCUGAN with Python CLI

```
> nix develop
> python main.py -h
usage: main.py [-h] [--scale {2,3,4}] [--no-pro] [--denoise {-1,0,1,2,3}] [--no-half] [--cache-mode {0,1,2,3}] [--tile TILE] [--alpha ALPHA]
               [--output OUTPUT] [--force] [--threads-per-card THREADS_PER_CARD] [--crf CRF] [--preset PRESET]
               input

Upscaler for anime images and videos

positional arguments:
  input                 Input path or directory (no recursive)

options:
  -h, --help            show this help message and exit
  --scale {2,3,4}       Upscaling factor
  --no-pro              Not using pro model
  --denoise {-1,0,1,2,3}
                        Denoising level. -1 for conservative.
  --no-half             Do not use half precision. Full precision is not recommended.
  --cache-mode {0,1,2,3}
                        Inference time: 0 < 1 < 2 < 3. VMem usage: 0 > 1 > 2 > 3.
  --tile TILE           Make (tile+1) x (tile+1) tiles for vmem usage reduction. 0 for no tiling.
  --alpha ALPHA         AI enhancement strength. The bigger the value, then the less adjustment, the less artifacts and the blurrier the result.
                        1 means no adjustment.
  --output OUTPUT       Output directory, default: <input-dir>/out
  --force               Overwrite existing output files

Video:
  --threads-per-card THREADS_PER_CARD
  --crf CRF             FFmpeg CRF value (lower is better quality)
  --preset PRESET       FFmpeg preset (e.g. slow, medium, fast, faster)
```

## References

- [RealCUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
