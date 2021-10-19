import os
import sys
from collections import deque, defaultdict
from typing import Dict, Union

sys.path.append("../../")

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):
        if "record_dir" not in cfg and "resume" not in cfg:
            raise ("The required parameter is not set.")
        # log_dir = os.path.join(pth.DATA_DIR, cfg.task, cfg.record_dir)
        log_dir = cfg.record_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not cfg.resume:
            os.system("rm -rf {}".format(log_dir))
        self.writer = SummaryWriter(log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)

    def update_image_stats(self, image_stats: Dict) -> None:
        """
        Arg:
            image_stats(Dict[batch_imgs]):
            辞書の内部に保存される値は、
            * 4D形状のミニバッチテンソル (B x C x H x W)
            * すべて同じサイズの画像のリスト。
        """
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def update_loss_stats(self, loss_dict: Dict) -> None:
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def record(
        self,
        prefix,
        step: int = -1,
        loss_stats: Union[Dict, None] = None,
        image_stats: Union[Dict, None] = None,
    ):
        pattern = prefix + "/{}"
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats
        image_stats = image_stats if image_stats else self.image_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        for k, v in self.image_stats.items():
            # RGB かつ [0, 1] の範囲の値を持つ場合
            if len(v.size()) == 3:
                b_size, h, w = v.size()[0], v.size()[1], v.size()[2]
                v = v.view(b_size, -1, h, w)

            v = v.float() if v.dtype != torch.float32 else v
            self.writer.add_image(pattern.format(k), vutils.make_grid(v), step)

        del loss_stats

    def state_dict(self):
        scalar_dict = {}
        scalar_dict["step"] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict["step"]

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append("{}: {:.4f}".format(k, v.avg))
        loss_state = "  ".join(loss_state)

        recording_state = "  ".join(
            ["epoch: {}", "step: {}", "{}", "batch_time: {:.3f} sec."]
        )
        return recording_state.format(
            self.epoch,
            self.step,
            loss_state,
            # self.data_time.avg,
            self.batch_time.avg,
        )


def make_recorder(cfg):
    return Recorder(cfg)
