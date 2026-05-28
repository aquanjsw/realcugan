from pathlib import Path

import numpy as np
import torch

from .UpCunet2x import UpCunet2x
from .UpCunet3x import UpCunet3x
from .UpCunet4x import UpCunet4x

DENOISE_DICT = {
    -1: "conservative",
    0: "on-denoise",
    1: "denoise1x",
    2: "denoise2x",
    3: "denoise3x",
}


class RealWaifuUpScaler:
    def __init__(self, scale, denoise, weights_dir: Path, device, half, pro):
        match scale:
            case 2:
                self.model = UpCunet2x()
            case 3:
                self.model = UpCunet3x()
            case 4:
                self.model = UpCunet4x()
            case _:
                raise ValueError(f"Unsupported scale: {scale}")
        self.model.to(device)
        if half:
            self.model.half()
        weights = self.load_weights(scale, denoise, weights_dir, device, pro)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self.pro = pro
        self.half = half
        self.device = device
        self.scale = scale
        self.setup_str = f"up{scale}x-{DENOISE_DICT[denoise]}"
        if pro:
            self.setup_str = f"pro-{self.setup_str}"

    @staticmethod
    def load_weights(scale, denoise, weights_dir: Path, device, pro):
        name = f"up{scale}x-{DENOISE_DICT[denoise]}.pth"
        if pro:
            name = f"pro-{name}"
        weights = torch.load(weights_dir / name, map_location=device)
        if pro:
            del weights["pro"]
        return weights

    def np2tensor(self, np_frame):
        if self.pro:
            if not self.half:
                return (
                    torch.from_numpy(np.transpose(np_frame, (2, 0, 1)))
                    .unsqueeze(0)
                    .to(self.device)
                    .float()
                    / (255 / 0.7)
                    + 0.15
                )
            else:
                return (
                    torch.from_numpy(np.transpose(np_frame, (2, 0, 1)))
                    .unsqueeze(0)
                    .to(self.device)
                    .half()
                    / (255 / 0.7)
                    + 0.15
                )
        else:
            if not self.half:
                return (
                    torch.from_numpy(np.transpose(np_frame, (2, 0, 1)))
                    .unsqueeze(0)
                    .to(self.device)
                    .float()
                    / 255
                )
            else:
                return (
                    torch.from_numpy(np.transpose(np_frame, (2, 0, 1)))
                    .unsqueeze(0)
                    .to(self.device)
                    .half()
                    / 255
                )

    def tensor2np(self, tensor):
        return np.transpose(tensor.squeeze().cpu().numpy(), (1, 2, 0))

    def __call__(self, x, tile_mode, cache_mode, alpha):
        with torch.no_grad():
            tensor = self.np2tensor(x)
            if cache_mode == 3:
                result = self.tensor2np(
                    self.model.forward_gap_sync(tensor, tile_mode, alpha, self.pro)
                )
            elif cache_mode == 2:
                result = self.tensor2np(
                    self.model.forward_fast_rough(tensor, tile_mode, alpha, self.pro)
                )
            else:
                result = self.tensor2np(
                    self.model(tensor, tile_mode, cache_mode, alpha, self.pro)
                )
        return result
