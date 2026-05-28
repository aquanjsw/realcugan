import os

import torch
import torch.nn.functional as F

from .UNets import UNet1x3, UNet2
from .UpCunetBase import UpCunetBase
from .utils import dq, q


class UpCunet3x(UpCunetBase):
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet3x, self).__init__()
        self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x, tile_mode, cache_mode, alpha, pro):
        n, c, h0, w0 = x.shape
        if "Half" in x.type():
            if_half = True
        else:
            if_half = False
        if tile_mode == 0:  # 不tile
            ph = ((h0 - 1) // 4 + 1) * 4
            pw = ((w0 - 1) // 4 + 1) * 4
            x = F.pad(
                x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect"
            )  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x, alpha)
            x = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x)
            if w0 != pw or h0 != ph:
                x = x[:, :, : h0 * 3, : w0 * 3]
            if pro:
                return ((x - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
            else:
                return (x * 255).round().clamp_(0, 255).byte()
        elif tile_mode == 1:  # 对长边减半
            if w0 >= h0:
                crop_size_w = (
                    (w0 - 1) // 8 * 8 + 8
                ) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_h = (h0 - 1) // 4 * 4 + 4  # 能被2整除
            else:
                crop_size_h = (
                    (h0 - 1) // 8 * 8 + 8
                ) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_w = (w0 - 1) // 4 * 4 + 4  # 能被2整除
            crop_size = (crop_size_h, crop_size_w)
        elif tile_mode >= 2:
            tile_mode = min(min(h0, w0) // 128, int(tile_mode))  # 最小短边为128*128
            t4 = tile_mode * 4
            crop_size = (
                ((h0 - 1) // t4 * t4 + t4) // tile_mode,
                ((w0 - 1) // t4 * t4 + t4) // tile_mode,
            )
        else:
            print("tile_mode config error")
            os._exit(233)
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect")
        n, c, h, w = x.shape
        if if_half:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        n_patch = 0
        tmp_dict = {}
        for i in range(0, h - 28, crop_size[0]):
            tmp_dict[i] = {}
            for j in range(0, w - 28, crop_size[1]):
                x_crop = x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                n, c1, h1, w1 = x_crop.shape
                tmp0, x_crop = self.unet1.forward_a(x_crop)
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        x_crop.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3), keepdim=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                tmp_dict[i][j] = (tmp0, x_crop)
        se_mean0 /= n_patch
        if if_half:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = tmp_dict[i][j]
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                opt_unet1 = F.pad(opt_unet1, (-20, -20, -20, -20))
                if cache_mode:
                    opt_unet1, tmp_x1 = q(opt_unet1, cache_mode), q(tmp_x1, cache_mode)
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x2.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3), keepdim=True)
                if cache_mode:
                    tmp_x2 = q(tmp_x2, cache_mode)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)
        se_mean1 /= n_patch
        if if_half:
            se_mean0 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2 = tmp_dict[i][j]
                if cache_mode:
                    tmp_x2 = dq(
                        tmp_x2[0], if_half, cache_mode, tmp_x2[1], tmp_x2[2], tmp_x2[3]
                    )
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                if cache_mode:
                    tmp_x2 = q(tmp_x2, cache_mode)
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x3.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3), keepdim=True)
                if cache_mode:
                    tmp_x3 = q(tmp_x3, cache_mode)
                se_mean0 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2, tmp_x3)
        se_mean0 /= n_patch
        if if_half:
            se_mean1 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_dict[i][j]
                if cache_mode:
                    tmp_x3 = dq(
                        tmp_x3[0], if_half, cache_mode, tmp_x3[1], tmp_x3[2], tmp_x3[3]
                    )
                assert self.unet2.conv3.seblock is not None
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(tmp_x3, se_mean0)
                if cache_mode:
                    tmp_x2 = dq(
                        tmp_x2[0], if_half, cache_mode, tmp_x2[1], tmp_x2[2], tmp_x2[3]
                    )
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3) * alpha
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x4.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3), keepdim=True)
                if cache_mode:
                    tmp_x4 = q(tmp_x4, cache_mode)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)
        se_mean1 /= n_patch
        res = torch.zeros(
            (n, c, h * 3 - 84, w * 3 - 84), dtype=torch.uint8, device=x.device
        )
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                x, tmp_x1, tmp_x4 = tmp_dict[i][j]
                if cache_mode:
                    tmp_x4 = dq(
                        tmp_x4[0], if_half, cache_mode, tmp_x4[1], tmp_x4[2], tmp_x4[3]
                    )
                assert self.unet2.conv4.seblock is not None
                tmp_x4 = self.unet2.conv4.seblock.forward_mean(tmp_x4, se_mean1)
                if cache_mode:
                    tmp_x1 = dq(
                        tmp_x1[0], if_half, cache_mode, tmp_x1[1], tmp_x1[2], tmp_x1[3]
                    )
                x0 = self.unet2.forward_d(tmp_x1, tmp_x4)
                if cache_mode:
                    x = dq(x[0], if_half, cache_mode, x[1], x[2], x[3])
                del tmp_dict[i][j]
                x = torch.add(x0, x)  # x0是unet2的最终输出
                if pro:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = ((x - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
                else:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = (x * 255).round().clamp_(0, 255).byte()
        del tmp_dict
        # torch.cuda.empty_cache()
        if w0 != pw or h0 != ph:
            res = res[:, :, : h0 * 3, : w0 * 3]
        return res

    def forward_gap_sync(self, x, tile_mode, alpha, pro):
        n, c, h0, w0 = x.shape
        if "Half" in x.type():
            if_half = True
        else:
            if_half = False
        if tile_mode == 0:  # 不tile
            ph = ((h0 - 1) // 4 + 1) * 4
            pw = ((w0 - 1) // 4 + 1) * 4
            x = F.pad(
                x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect"
            )  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x, alpha)
            x = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x)
            if w0 != pw or h0 != ph:
                x = x[:, :, : h0 * 3, : w0 * 3]
            if pro:
                return ((x - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
            else:
                return (x * 255).round().clamp_(0, 255).byte()
        elif tile_mode == 1:  # 对长边减半
            if w0 >= h0:
                crop_size_w = (
                    (w0 - 1) // 8 * 8 + 8
                ) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_h = (h0 - 1) // 4 * 4 + 4  # 能被2整除
            else:
                crop_size_h = (
                    (h0 - 1) // 8 * 8 + 8
                ) // 2  # 减半后能被2整除，所以要先被4整除
                crop_size_w = (w0 - 1) // 4 * 4 + 4  # 能被2整除
            crop_size = (crop_size_h, crop_size_w)
        elif tile_mode >= 2:
            tile_mode = min(min(h0, w0) // 128, int(tile_mode))  # 最小短边为128*128
            t4 = tile_mode * 4
            crop_size = (
                ((h0 - 1) // t4 * t4 + t4) // tile_mode,
                ((w0 - 1) // t4 * t4 + t4) // tile_mode,
            )
        else:
            print("tile_mode config error")
            os._exit(233)
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect")
        n, c, h, w = x.shape
        if if_half:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        n_patch = 0
        h1, w1 = crop_size[0] + 28, crop_size[1] + 28
        ######stage1
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        x_crop.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3), keepdim=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
        se_mean0 /= n_patch
        ######stage1+state2
        if if_half:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x2.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3), keepdim=True)
                se_mean1 += tmp_se_mean
        se_mean1 /= n_patch
        ######stage1+state2+state3
        if if_half:
            se_mean2 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean2 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x3.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3), keepdim=True)
                se_mean2 += tmp_se_mean
        se_mean2 /= n_patch
        #########stage1+state2+state3+stage4
        if if_half:
            se_mean3 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean3 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                assert self.unet2.conv3.seblock is not None
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(tmp_x3, se_mean2)
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3)
                assert self.unet2.conv4.seblock is not None
                if if_half:  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(
                        tmp_x4.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3), keepdim=True)
                se_mean3 += tmp_se_mean
        se_mean3 /= n_patch
        ###########stage1+state2+state3+stage4+stage_tail
        res = torch.zeros(
            (n, c, h * 3 - 84, w * 3 - 84), dtype=torch.uint8, device=x.device
        )
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                x_crop = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(x_crop)
                x_crop = F.pad(x_crop, (-20, -20, -20, -20))
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                assert self.unet2.conv3.seblock is not None
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(tmp_x3, se_mean2)
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3) * alpha
                assert self.unet2.conv4.seblock is not None
                tmp_x4 = self.unet2.conv4.seblock.forward_mean(tmp_x4, se_mean3)
                x0 = self.unet2.forward_d(tmp_x1, tmp_x4)
                x_crop = torch.add(x0, x_crop)
                if pro:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = ((x_crop - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
                else:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = (x_crop * 255.0).round().clamp_(0, 255).byte()
        # torch.cuda.empty_cache()
        if w0 != pw or h0 != ph:
            res = res[:, :, : h0 * 3, : w0 * 3]
        return res

    def forward_fast_rough(self, x, tile_mode, alpha, pro):  # 1.7G
        n, c, h0, w0 = x.shape
        if "Half" in x.type():
            if_half = True
        else:
            if_half = False
        if tile_mode < 3:
            return self.forward(x, tile_mode, 1, alpha, pro)  # 至少切成3x3
        elif tile_mode >= 3:
            tile_mode = min(min(h0, w0) // 128, int(tile_mode))  # 最小短边为128*128
            if tile_mode < 3:
                return self.forward(x, tile_mode, 1, alpha, pro)
            t4 = tile_mode * 4
            crop_size = (
                ((h0 - 1) // t4 * t4 + t4) // tile_mode,
                ((w0 - 1) // t4 * t4 + t4) // tile_mode,
            )  # 5.6G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), "reflect")
        n, c, h, w = x.shape
        h1, w1 = crop_size[0] + 28, crop_size[1] + 28
        n_patch = 0
        ###########stage1+state2+state3+stage4
        if if_half:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean0 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        if if_half:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean1 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        if if_half:
            se_mean2 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean2 = torch.zeros((n, 128, 1, 1), device=x.device, dtype=torch.float32)
        if if_half:
            se_mean3 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float16)
        else:
            se_mean3 = torch.zeros((n, 64, 1, 1), device=x.device, dtype=torch.float32)
        for i in range(0, h - 28, crop_size[0]):
            if (i // crop_size[0]) % 2 == 0:
                continue
            for j in range(0, w - 28, crop_size[1]):
                if (j // crop_size[1]) % 2 == 0:
                    continue
                n_patch += 1
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                if if_half:
                    se_mean0 += torch.mean(
                        x_crop.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    se_mean0 += torch.mean(x_crop, dim=(2, 3), keepdim=True)
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(
                    x_crop, se_mean0 / n_patch
                )
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                if if_half:
                    se_mean1 += torch.mean(
                        tmp_x2.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    se_mean1 += torch.mean(tmp_x2, dim=(2, 3), keepdim=True)
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(
                    tmp_x2, se_mean1 / n_patch
                )
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                if if_half:
                    se_mean2 += torch.mean(
                        tmp_x3.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    se_mean2 += torch.mean(tmp_x3, dim=(2, 3), keepdim=True)
                assert self.unet2.conv3.seblock is not None
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(
                    tmp_x3, se_mean2 / n_patch
                )
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3)
                if if_half:
                    se_mean3 += torch.mean(
                        tmp_x4.float(), dim=(2, 3), keepdim=True
                    ).half()
                else:
                    se_mean3 += torch.mean(tmp_x4, dim=(2, 3), keepdim=True)
        # print("3x-n_patch=%s,tile_mode=%s" % (n_patch,tile_mode))
        ###########stage1+state2+state3+stage4+stage_tail
        res = torch.zeros(
            (n, c, h * 3 - 84, w * 3 - 84), dtype=torch.uint8, device=x.device
        )
        for i in range(0, h - 28, crop_size[0]):
            for j in range(0, w - 28, crop_size[1]):
                tmp0, x_crop = self.unet1.forward_a(
                    x[:, :, i : i + crop_size[0] + 28, j : j + crop_size[1] + 28]
                )
                assert self.unet1.conv2.seblock is not None
                x_crop = self.unet1.conv2.seblock.forward_mean(
                    x_crop, se_mean0 / n_patch
                )
                x_crop = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(x_crop)
                x_crop = F.pad(x_crop, (-20, -20, -20, -20))
                assert self.unet2.conv2.seblock is not None
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(
                    tmp_x2, se_mean1 / n_patch
                )
                tmp_x2, tmp_x3 = self.unet2.forward_b(tmp_x2)
                assert self.unet2.conv3.seblock is not None
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(
                    tmp_x3, se_mean2 / n_patch
                )
                assert self.unet2.conv4.seblock is not None
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3) * alpha
                tmp_x4 = self.unet2.conv4.seblock.forward_mean(
                    tmp_x4, se_mean3 / n_patch
                )
                x0 = self.unet2.forward_d(tmp_x1, tmp_x4)
                x_crop = torch.add(x0, x_crop)
                if pro:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = ((x_crop - 0.15) * (255 / 0.7)).round().clamp_(0, 255).byte()
                else:
                    res[
                        :, :, i * 3 : i * 3 + h1 * 3 - 84, j * 3 : j * 3 + w1 * 3 - 84
                    ] = (x_crop * 255.0).round().clamp_(0, 255).byte()
        # torch.cuda.empty_cache()
        if w0 != pw or h0 != ph:
            res = res[:, :, : h0 * 3, : w0 * 3]
        return res
