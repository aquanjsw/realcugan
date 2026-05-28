import argparse
from pathlib import Path

from loguru import logger

parser = argparse.ArgumentParser(description="Upscaler for anime images and videos")
parser.add_argument(
    "--scale", default=3, type=int, choices=(2, 3, 4), help="Upscaling factor"
)
parser.add_argument("--no-pro", action="store_true", help="Not using pro model")
parser.add_argument(
    "--denoise",
    default=3,
    type=int,
    choices=(-1, 0, 1, 2, 3),
    help="Denoising level. -1 for conservative.",
)
parser.add_argument(
    "--no-half",
    action="store_true",
    help="Do not use half precision. You may not need this.",
)
parser.add_argument(
    "--cache-mode",
    default=0,
    type=int,
    choices=(0, 1, 2, 3),
    help="Inference time: 0 < 1 < 2 < 3. VMem usage: 0 > 1 > 2 > 3.",
)
parser.add_argument(
    "--tile",
    default=0,
    type=int,
    help="Make (tile+1) x (tile+1) tiles for vmem usage reduction. 0 for no tiling.",
)
parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="AI enhancement strength. The bigger the value, then the less adjustment"
    ", the less artifacts and the blurrier the result. 1 means no adjustment.",
)
parser.add_argument("--output", help="Output directory, default: <input-dir>/out")
parser.add_argument(
    "--force", action="store_true", help="Overwrite existing output files"
)
parser.add_argument("input", help="Input path or directory (no recursive)")

video = parser.add_argument_group("video")
video.add_argument("--threads-per-card", default=2)
video.add_argument(
    "--crf", default=21, help="FFmpeg CRF value (lower is better quality)"
)
video.add_argument(
    "--preset", default="medium", help="FFmpeg preset (e.g. slow, medium, fast, faster)"
)


def get_output(indir, output) -> Path:
    if not output:
        output = indir / "out"
    else:
        output = Path(output)
        if output.is_file():
            logger.error(f"Output {output} is a file, but should be a directory")
            exit(1)
        if output == indir:
            logger.error(f"Output {output} is the same as input directory")
            exit(1)
    return output


args = parser.parse_args()

args.input = Path(args.input)
if args.input.is_file():
    args.output = get_output(args.input.parent, args.output)
    args.input = [args.input]
elif args.input.is_dir():
    args.output = get_output(args.input, args.output)
    args.input = list(filter(lambda x: x.is_file(), args.input.iterdir()))

args.output.mkdir(parents=True, exist_ok=True)

args.half = not args.no_half
args.pro = not args.no_pro
