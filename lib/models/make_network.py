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
        cfg (CfgNode): `config` 情報が保存された辞書．

    Returns:
        [type]: [description]
    """
    if "model" not in cfg and "network" not in cfg and "train_type" not in cfg:
        raise ("The required parameter for `make_network` is not set.")
    net_name = cfg.network
    path = os.path.join(pth.LIB_DIR, "models", net_name, "__init__.py")
    spec = importlib.util.spec_from_file_location("__init__", path)
    get_network_fun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(get_network_fun)
    network = get_network_fun.get_network(cfg)
    if cfg.train_type == "transfer":
        if int(cfg.num_classes) > 0:
            network = transfer_network(network, cfg.num_classes)
        else:
            raise ValueError("Invalid value for num_classes")

    return network


def transfer_network(network, num_classes: int):
    """
    転移学習用にモデルの全結合層を未学習のものに置き換える関数

    Args:
        network: make_network で読みだされたモデル
        num_classes (int): 転移学習時のクラス数

    Return:
        network
    """
    if hasattr(network, "classifier"):
        # EfficientNet の場合
        num_ftrs = network.classifier.in_features
        fc = nn.Linear(num_ftrs, num_classes)
        network.classifier = fc
    elif hasattr(network, "fc"):
        # ResNet の場合
        num_ftrs = network.fc.in_features
        fc = nn.Linear(num_ftrs, num_classes)
        network.classifier = fc
    return network


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.train = CfgNode()
    # cfg.network = res
    # cfg.model = "res_152"
    # cfg.train_type = "transfer"
    cfg.network = "smp"
    cfg.model = "efficientnet-b0"
    cfg.num_classes = 2

    model = make_network(cfg)
    for arc in model:
        print(arc)
