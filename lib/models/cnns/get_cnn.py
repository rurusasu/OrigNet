import sys
from typing import Literal

sys.path.append(".")
sys.path.append("../../../")

import torch

from lib.models.cnns.alexnet import get_alex_net as get_alex
from lib.models.cnns.efficientnet import get_efficient_net as get_eff
from lib.models.cnns.inception import get_inception_net as get_inc
from lib.models.cnns.resnet import get_res_net as get_res
from lib.models.cnns.vgg import get_vgg_net as get_vgg

_network_factory = {
    "alex": get_alex,
    "eff": get_eff,
    "inc": get_inc,
    "res": get_res,
    "vgg": get_vgg,
}


def GetCNN(
    model_name: str,
    num_classes: int,
    encoder_name: str,
    train_type: Literal["scratch", "transfer"] = "scratch",
) -> torch.nn:
    """
    CNN の構造を読み出す関数．

    Args:
        model_name (str): 読み出したい CNN の構造名．
        num_classes (int): 出力数．
        encoder_name (str):
            エンコーダに用いるモデル構造名．
            cnn では使用しないダミー変数．
        train_type (Literal["scratch", "transfer"], optional):
            訓練のタイプ.
            `scratch`: 重みを初期化して読み出す．
            `transfer`: 転移学習用のモデルを読み出す．
            Defaults to "scratch".

    Raises:
        ValueError: The specified `model_name` does not exist.
        ValueError: The `num_classes` must be of type int and `num_classes` > 0.
        ValueError: For train_type, select scratch or transfer.

    Returns:
        torch.nn: CNN の構造
    """

    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("The num_classes must be of type int and num_classes > 0.")

    if train_type != "scratch" and train_type != "transfer":
        raise ValueError("For train_type, select scratch or transfer.")

    model_num = -1
    arch = model_name
    if "_" in arch:
        model_num = str(arch[arch.find("_") + 1 :]) if "_" in arch else 0
        arch = arch[: arch.find("_")]

    if arch not in _network_factory:
        raise ValueError(f"The specified {arch} does not exist.")
    get_model = _network_factory[arch]

    # 転移学習を行う場合
    if train_type == "transfer":
        model = get_model(model_num, pretrained=True)
    elif train_type == "scratch" and num_classes > 0:
        model = get_model(model_num, pretrained=False, num_classes=num_classes)

    return model
