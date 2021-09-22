import importlib.machinery
import importlib.util
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import torch.nn as nn
from yacs.config import CfgNode

from lib.config.config import pth


def make_network(cfg: CfgNode):
    """Network を読みだす関数

    Args:
        cfg (CfgNode):

    Returns:
        [type]: [description]
    """
    if "model" not in cfg and "network" not in cfg and "train" not in cfg:
        raise ("The required parameter for `make_network` is not set.")
    name = cfg.model
    if name.find("_"):
        name = name[: name.find("_")]
    path = os.path.join(pth.LIB_DIR, "models", name, "__init__.py")
    # spec = importlib.util.spec_from_file_location("get_network", path)
    spec = importlib.util.spec_from_file_location("__init__", path)
    get_network_fun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(get_network_fun)
    model = get_network_fun.get_network(cfg)
    if cfg.network == "transfer":
        model = transfer_network(model, cfg.num_classes)
    return model


def transfer_network(model, num_classes: int):
    """
    転移学習用にモデルの全結合層を未学習のものに置き換える関数

    Args:
        model:
        num_classes (int): 転移学習時のクラス数
    """
    # ネットワークに全結合層が存在する場合
    if model.fc:
        num_ftrs = model.fc.in_features
        fc = nn.Linear(num_ftrs, num_classes)
        model.fc = fc
    return model


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.model = "res_152"
    cfg.train = CfgNode()
    cfg.train.pretrained = False

    model = make_network(cfg)
    for arc in model:
        print(arc)
