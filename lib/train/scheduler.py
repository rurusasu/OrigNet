from bisect import bisect_right
from collections import Counter

import torch
from torch.optim.lr_scheduler import StepLR
from yacs.config import CfgNode


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones: tuple,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


def make_lr_scheduler(cfg: CfgNode, optimizer):
    if (
        "train" not in cfg
        and "scheduler" not in cfg.train
        and "milestones" not in cfg.train
        and "warp_iter" not in cfg.train
        and "gamma" not in cfg.train
    ):
        raise ("The required parameter for `make_lr_scheduler` is not set.")
    if cfg.train.scheduler == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=cfg.train.milestones,
            gamma=cfg.train.gamma,
            warmup_factor=1.0 / 3,
            warmup_iters=cfg.train.warp_iter,
            warmup_method="linear",
        )
    elif cfg.train.scheduler == "multi_step_lr":
        scheduler = MultiStepLR(
            optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma
        )
    elif cfg.train.scheduler == "step_lr":
        scheduler = StepLR(
            optimizer, gamma=cfg.train.gamma, step_size=cfg.train.warp_iter
        )
    else:
        raise ("The required parameter for `LR Scheduler` is not set.")
    return scheduler


if __name__ == "__main__":

    import torch.nn as nn
    import torch.optim as optim
    from matplotlib import pyplot as plt

    cfg = CfgNode()
    cfg.train = CfgNode()
    cfg.train.scheduler = "multi_step_lr"
    cfg.train.milestones = (20, 40, 60, 80)
    cfg.train.gamma = 0.5

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01, momentum=0.9)

    scheduler = make_lr_scheduler(cfg, optimizer)
    data_x = []
    data_y = []
    for ep in range(0, 100):
        scheduler.step()

        data_x.append(ep)
        data_y.append(optimizer.param_groups[0]["lr"])

    plt.plot(data_x, data_y)

    plt.title("Learning Rate Schedule")
    plt.xlabel("epoches")
    plt.ylabel("Learning Rate")

    plt.show()
