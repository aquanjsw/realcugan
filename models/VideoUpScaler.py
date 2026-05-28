import threading
from pathlib import Path
from random import uniform
from tempfile import NamedTemporaryFile
from time import sleep
from time import time as ttime

import torch
from loguru import logger
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from torch.multiprocessing import Queue
from tqdm import tqdm

from .RealWaifuUpScaler import RealWaifuUpScaler


class UpScalerMT(threading.Thread):
    def __init__(
        self, inp_q, res_q, device, model, p_sleep, nt, tile, cache_mode, alpha
    ):
        threading.Thread.__init__(self)
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.cache_mode = cache_mode
        self.nt = nt
        self.p_sleep = p_sleep
        self.tile = tile
        self.alpha = alpha

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame, self.tile, self.cache_mode, self.alpha)
        if self.nt > 1:
            sleep(uniform(self.p_sleep[0], self.p_sleep[1]))
        return (idx, res)

    def run(self):
        while 1:
            tmp = self.inp_q.get()
            if not tmp:
                # print("exit")
                break
            self.res_q.put(self.inference(tmp))


class VideoUpScaler:
    DECODE_SLEEP = 0.002
    P_SLEEP = (0.005, 0.012)

    def __init__(
        self,
        models: list[RealWaifuUpScaler],
        tile,
        cache_mode,
        alpha,
        outdir: Path,
        threads_per_card,
        crf,
        preset,
        force,
    ):
        self.models = models
        self.tile = tile
        self.cache_mode = cache_mode
        self.alpha = alpha
        self.outdir = outdir
        self.force = force
        self.crf = crf
        self.preset = preset
        self.scale = self.models[0].scale
        self.encode_params = (
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
        )
        self.threads_per_card = threads_per_card
        self.ngpu = len(models)

        self.inp_q = Queue(threads_per_card * self.ngpu * 2)  # 抽帧缓存上限帧数
        self.res_q = Queue(threads_per_card * self.ngpu * 2)  # 超分帧结果缓存上限
        for model in self.models:
            for _ in range(self.threads_per_card):
                upscaler = UpScalerMT(
                    self.inp_q,
                    self.res_q,
                    model.device,
                    model,
                    self.P_SLEEP,
                    self.threads_per_card,
                    tile,
                    cache_mode,
                    alpha,
                )
                upscaler.start()

    def __call__(self, inpath: Path):
        assert inpath.is_file()

        setup_str = "-".join(
            (
                f"crf{self.crf}",
                f"{self.preset}",
                f"tile{self.tile}",
                f"cache{self.cache_mode}",
                f"alpha{self.alpha}",
            )
        )
        name = f"{inpath.stem}-{self.models[0].setup_str}-{setup_str}.mp4"
        outpath = self.outdir / name

        if outpath.exists() and not self.force:
            logger.info(f"Video already exists, skipping: {outpath}")
            raise FileExistsError

        # OSError for wrong format
        clip = VideoFileClip(inpath)

        if not clip.reader:
            logger.error(inpath, "failed to load, no reader")
            raise OSError

        total_frame = clip.reader.n_frames

        w, h = clip.reader.size
        fps = clip.reader.fps
        if clip.audio:
            tmpfile = NamedTemporaryFile(suffix=".m4a", delete=False)
            tmpfile.close()
            tmp_audio_path = tmpfile.name
            clip.audio.write_audiofile(tmp_audio_path, codec="aac")
            writer = FFMPEG_VideoWriter(
                outpath.as_posix(),
                (w * self.scale, h * self.scale),
                fps,
                ffmpeg_params=self.encode_params,
                audiofile=tmp_audio_path,
            )  # slower#medium
        else:
            writer = FFMPEG_VideoWriter(
                outpath.as_posix(),
                (w * self.scale, h * self.scale),
                fps,
                ffmpeg_params=self.encode_params,
            )  # slower#medium
        now_idx = 0
        idx2res = {}
        t0 = ttime()
        for idx, frame in tqdm(enumerate(clip.iter_frames()), total=total_frame):
            while 1:  # 带重试的入队：若inp_q满则持续排空res_q直到有空位
                while not self.res_q.empty():
                    iidx, res = self.res_q.get()
                    idx2res[iidx] = res
                try:
                    self.inp_q.put((idx, frame), block=True, timeout=0.05)
                    break
                except Exception:
                    pass  # inp_q仍满，继续排空res_q再重试
            sleep(
                self.DECODE_SLEEP
            )  # 否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
            # if (idx % 100 == 0):
            while 1:  # 按照idx排序写帧
                if now_idx not in idx2res:
                    break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
        idx += 1
        while 1:
            # print(2,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
            # if (now_idx >= idx + 1): break  # 全部帧都写入了，跳出
            while 1:  # 取出处理好的所有结果
                if self.res_q.empty():
                    break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            while 1:  # 按照idx排序写帧
                if now_idx not in idx2res:
                    break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
            if self.inp_q.qsize() == 0 and self.res_q.qsize() == 0 and idx == now_idx:
                break
            sleep(0.02)
        # print(3,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
        writer.close()
        t1 = ttime()
        logger.info(f"Video saved: {outpath}, time: {t1 - t0:.2f}s")

    def __del__(self):
        for _ in range(self.threads_per_card * self.ngpu):
            self.inp_q.put(None)
