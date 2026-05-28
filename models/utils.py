def q(inp, cache_mode):
    maxx = inp.max()
    minn = inp.min()
    delta = maxx - minn
    if cache_mode == 2:
        return (
            ((inp - minn) / delta * 255).round().byte().cpu(),
            delta,
            minn,
            inp.device,
        )  # 大概3倍延时#太慢了，屏蔽该模式
    elif cache_mode == 1:
        return (
            ((inp - minn) / delta * 255).round().byte(),
            delta,
            minn,
            inp.device,
        )  # 不用CPU转移


def dq(inp, if_half, cache_mode, delta, minn, device):
    if cache_mode == 2:
        if if_half:
            return inp.to(device).half() / 255 * delta + minn
        else:
            return inp.to(device).float() / 255 * delta + minn
    elif cache_mode == 1:
        if if_half:
            return inp.half() / 255 * delta + minn  # 不用CPU转移
        else:
            return inp.float() / 255 * delta + minn
