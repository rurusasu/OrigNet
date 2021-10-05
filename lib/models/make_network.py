from operator import ne
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import torch.nn as nn
from yacs.config import CfgNode

from lib.models.cnns.get_cnn import GetCNN
from lib.models.smp.get_semantic_segm import GetSemanticSegm


_network_factory = {
    "cnns": GetCNN,
    "smp": GetSemanticSegm,
}


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
    get_network_fun = _network_factory[net_name]
    network = get_network_fun(cfg)

    if net_name == "cnns":
        if cfg.train_type == "transfer":
            if int(cfg.num_classes) > 0:
                network = transfer_network(network, cfg.num_classes)
            else:
                raise ValueError("Invalid value for num_classes")
    elif net_name == "smp":
        pass

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
        # AlexNet の場合
        if issubclass(type(network.classifier), nn.Sequential):
            # num_ftrs = network.classifier[-1].in_features
            num_ftrs = network.classifier[1].in_features
            out_ftrs = network.classifier[1].out_features
            fc = nn.Linear(num_ftrs, out_features=out_ftrs)
            network.classifier[1] = fc
            num_ftrs = network.classifier[4].in_features
            out_ftrs = network.classifier[4].out_features
            fc = nn.Linear(num_ftrs, out_features=out_ftrs)
            network.classifier[4] = fc
            num_ftrs = network.classifier[6].in_features
            out_ftrs = network.classifier[6].out_features
            fc = nn.Linear(num_ftrs, out_features=num_classes)
            network.classifier[6] = fc
        else:
            # EfficientNet の場合
            num_ftrs = network.classifier.in_features
            fc = nn.Linear(num_ftrs, out_features=num_classes)
            network.classifier = fc
    elif hasattr(network, "fc"):
        # ResNet の場合
        num_ftrs = network.fc.in_features
        fc = nn.Linear(num_ftrs, out_features=num_classes)
        network.fc = fc
    return network


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.train = CfgNode()
    # cfg.network = "cnns"
    # cfg.model = "res_152"
    # cfg.train_type = "transfer"
    cfg.network = "smp"
    cfg.model = "unetpp"
    cfg.encoder_name = "efficientnet-b0"
    cfg.num_classes = 2

    model = make_network(cfg)
    if cfg.network == "cnns":
        for arc in model:
            print(arc)
