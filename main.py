from pathlib import Path

import torch
from loguru import logger

from args import args
from models.ImageUpScaler import ImageUpScaler
from models.RealWaifuUpScaler import RealWaifuUpScaler
from models.VideoUpScaler import VideoUpScaler

CUR_DIR = Path(__file__).parent
WEIGHTS_DIR = CUR_DIR / "weights"

if torch.cuda.is_available():
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    logger.info(f"Found {len(devices)} CUDA devices.")
else:
    logger.info("CUDA is not available. Using CPU.")
    logger.warning("Video upscaling will be disabled.")
    devices = ["cpu"]

models = [
    RealWaifuUpScaler(
        args.scale, args.denoise, WEIGHTS_DIR, device, args.half, args.pro
    )
    for device in devices
]
image_upscaler = ImageUpScaler(
    models[0], args.tile, args.cache_mode, args.alpha, args.output, args.force
)
video_upscaler = VideoUpScaler(
    models,
    args.tile,
    args.cache_mode,
    args.alpha,
    args.output,
    args.threads_per_card,
    args.crf,
    args.preset,
    args.force,
)

for input in args.input:
    # The procedure matters
    # Image first so that the video upscaler doesn't waste time trying to
    # process an image file
    try:
        image_upscaler(input)
    except FileExistsError:
        continue
    except ValueError:
        try:
            video_upscaler(input)
        except OSError:
            logger.error(f"Failed to process {input}. Skipping.")
            continue
    except Exception as e:
        logger.error(f"An error occurred while processing {input}: {e}")
        continue

del video_upscaler
