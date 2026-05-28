from pathlib import Path

import cv2
from loguru import logger

from .RealWaifuUpScaler import RealWaifuUpScaler


class ImageUpScaler:
    def __init__(
        self, model: RealWaifuUpScaler, tile, cache_mode, alpha, outdir: Path, force
    ):
        self.model = model
        self.tile = tile
        self.cache_mode = cache_mode
        self.alpha = alpha
        self.outdir = outdir
        self.force = force

    def __call__(self, inpath: Path):
        assert inpath.is_file(), f"Input path is not a file: {inpath}"

        EXT = ".png"
        setup_str = f"tile{self.tile}-cache{self.cache_mode}-alpha{self.alpha}"
        name = f"{inpath.stem}-{self.model.setup_str}-{setup_str}{EXT}"
        outpath = self.outdir / name
        if outpath.exists() and not self.force:
            logger.info(f"Image already exists, skipping: {outpath}")
            raise FileExistsError

        x = cv2.imread(inpath)
        if x is None:
            raise ValueError
        x = x[:, :, [2, 1, 0]]
        x = self.model(x, self.tile, self.cache_mode, self.alpha)[:, :, ::-1]

        cv2.imencode(EXT, x)[1].tofile(outpath)
        logger.info(f"Image saved: {outpath}")
