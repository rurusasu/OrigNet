import sys

sys.path.append(".")
sys.path.append("../../../")

import torch.nn as nn

from lib.models.res.resnet import get_res_net as get_res

_network_factory = {
    "res": get_res,
}


def get_network(cfg):
    # from .resnet import get_res_net as get_res

    # _network_factory = {
    #    "res": get_res,
    # }
    arch = cfg.model
    if arch.find("_"):
        num_layers = int(arch[arch.find("_") + 1 :]) if "_" in arch else 0
        arch = arch[: arch.find("_")]
    else:
        raise ValueError(f"The specified cfg.network={arch} does not exist.")

    if arch not in _network_factory and "num_classes" not in cfg.num_classes:
        raise ValueError(f"The specified cfg.network={arch} does not exist.")
    get_model = _network_factory[arch]

    # 転移学習を行う場合
    if "pretrained" in cfg.train and cfg.train.pretrained:
        model = get_model(num_layers, pretrained=True, num_classes=cfg.num_classes)
    else:
        model = get_model(num_layers, pretrained=False, num_classes=cfg.num_classes)

    return model
