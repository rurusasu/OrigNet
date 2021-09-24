import torch
from lib.utils.optimizer.radam import RAdam

_optimizer_factory = {"adam": torch.optim.Adam, "radam": RAdam, "sgd": torch.optim.SGD}


def make_optimizer(cfg, network):
    if (
        "train" not in cfg
        and "lr" not in cfg.train
        and "weight_decay" not in cfg.train
        and "optim" not in cfg.train
    ):
        raise ("The required parameter for `make_optimizer` is not set.")
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay

    # 層ごとにハイパ－パラメタを設定する
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if "adam" in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](
            params, lr, weight_decay=weight_decay
        )
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
