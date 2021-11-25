import sys
from typing import Literal

sys.path.append(".")
sys.path.append("../../../")

import torch
from yacs.config import CfgNode

from lib.models.smp.unetpp import GetUNetPP as get_unet_pp

_network_factory = {"unetpp": get_unet_pp}


def GetSemanticSegm(
    model_name: str,
    num_classes: int,
    encoder_name: str,
    train_type: Literal["scratch", "transfer"] = "scratch",
) -> torch.nn:
    """
    セマンティックセグメンテーション用のモデル構造を読み出す関数．

    Args:
        model_name (str): 読み出したいモデルの構造名．
        num_classes (int): 出力数．
        encoder_name (str): エンコーダに用いるモデル構造名．
        train_type (Literal["scratch", "transfer"], optional):
            訓練のタイプ.
            `scratch`: 重みを初期化して読み出す．
            `transfer`: 転移学習用のモデルを読み出す．
            Defaults to "scratch".

    Raises:
        ValueError: The specified `model_name` does not exist.
        ValueError: The `encoder_name` must be of type str.
        ValueError: The `num_classes` must be of type int and `num_classes` > 0.
        ValueError: For train_type, select scratch or transfer.

    Returns:
        torch.nn: モデルの構造
    """
    if model_name not in _network_factory:
        raise ValueError(f"The specified {model_name} does not exist.")

    if not isinstance(encoder_name, str) and encoder_name != "":
        raise ValueError("The `encoder_name` must be of type str.")

    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("The num_classes must be of type int and num_classes > 0.")

    if train_type != "scratch" and train_type != "transfer":
        raise ValueError("For train_type, select scratch or transfer.")

    get_model = _network_factory[model_name]
    model = get_model(encoder_name=encoder_name, num_classes=num_classes)

    return model


if __name__ == "__main__":
    from lib.datasets.make_datasets import make_data_loader

    cfg = CfgNode()
    cfg.task = "semantic_segm"
    cfg.cls_names = ["laptop", "tv"]
    cfg.num_classes = len(cfg.cls_names)
    cfg.img_width = 224
    cfg.img_height = 224
    cfg.task = "semantic_segm"
    cfg.network = "smp"
    cfg.model = "unetpp"
    cfg.encoder_name = "resnet18"
    cfg.train = CfgNode()
    cfg.train.dataset = "COCO2017Val"
    cfg.train.batch_size = 4
    cfg.train.num_workers = 2
    cfg.train.batch_sampler = ""

    model = GetSemanticSegm(cfg)

    dloader = make_data_loader(cfg, is_train=True)
    for iter, batch in enumerate(dloader):
        img = batch["img"].float()
        output = model.forward(img)
