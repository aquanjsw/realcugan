import torch.nn as nn


class UpCunetBase(nn.Module):
    def forward_gap_sync(self, x, tile_mode, alpha, pro):
        pass

    def forward_fast_rough(self, x, tile_mode, alpha, pro):
        pass
