import sys
from typing import Literal, Tuple

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


def make_network(
    model_name: str,
    num_classes: int,
    network_name: Literal["cnns", "smp"],
    encoder_name: str,
    replaced_layer_num: int = 1,
    train_type: Literal["scratch", "transfer"] = "scratch",
):
    """Network を読みだす関数

    Args:
        model_name (str): 読み出したいモデルの構造名．
        num_classes (int): 出力数．
        encoder_name (str): エンコーダに用いるモデル構造名．
        replaced_layer_num (int, optional):
            転移学習時にモデルのパラメタを初期化する層の番号.
            出力層から1番目とカウント.
            Default to 1.
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
    if network_name not in _network_factory:
        raise ValueError(f"The specified {network_name} does not exist.")

    if not isinstance(encoder_name, str) and encoder_name != "":
        raise ValueError("The `encoder_name` must be of type str.")

    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("The num_classes must be of type int and num_classes > 0.")

    if train_type != "scratch" and train_type != "transfer":
        raise ValueError("For train_type, select scratch or transfer.")

    arch = {
        "model_name": model_name,
        "num_classes": num_classes,
        "encoder_name": encoder_name,
        "train_type": train_type,
    }
    get_network_fun = _network_factory[network_name]
    network = get_network_fun(**arch)

    if network_name == "cnns":
        # 転移学習の場合
        if train_type == "transfer":
            if replaced_layer_num > 1:
                network, _ = transfer_network(
                    network=network,
                    num_classes=num_classes,
                    replaced_layer_num=replaced_layer_num,
                )
            else:
                raise (
                    "Required parameters `replace_layer_num` for make_network are not set."
                )
    elif network_name == "smp":
        pass

    return network


def transfer_network(
    network, num_classes: int, replaced_layer_num: int = 1
) -> Tuple[nn.Sequential, int]:
    """
    転移学習用にモデルの全結合層を未学習のものに置き換える関数

    Args:
        network: make_network で読みだされたモデル
        num_classes (int): 転移学習時のクラス数．
        replaced_layer_num (int, optional): 置き換えたい全結合層の数．出力層のみ転移学習時のクラス数に置き換え，それ以外は weight を xavier の初期値で，bias を [0, 1] の一様分布の乱数で初期化する．Default to 1.

    Return:
        network (torch.nn.Sequential): 出力数と必要に応じて複数の全結合層の重みが初期化されたネットワーク．
        replaced_layer_num (int): 実際に重みが初期化された層の数．
    """
    if num_classes < 1:
        raise (
            "num_classes error. This value should be given as a positive integer greater than zero."
        )
    if replaced_layer_num < 1:
        raise (
            "relpaced layer number error. This value should be given as a positive integer greater than zero."
        )

    # 最終出力層の重みを初期化したかを判定するフラグ
    # False: 初期化した。
    out_layer_replaced = True
    iter = 0
    if hasattr(network, "classifier"):
        # classifier が複数の層から構成されている場合（例: AlexNet, VGG）
        if issubclass(type(network.classifier), nn.Sequential):
            for key, layer in reversed(network.classifier._modules.items()):
                if issubclass(type(layer), nn.Linear):
                    # 置き換えたい全結合層の数と実際に置き換えた全結合層の数が同数になった場合，break.
                    if iter > replaced_layer_num:
                        break
                    if out_layer_replaced:
                        in_features = layer.in_features  # 全結合層への既存の入力サイズを取得
                        fc = nn.Linear(
                            in_features, out_features=num_classes
                        )  # 訓練させたいクラス数に応じて出力サイズを変更
                        network.classifier._modules[key] = fc  # 全結合層を置き換え
                        out_layer_replaced = False  # フラグを変更
                        iter += 1
                    else:
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.uniform_(layer.bias, 0, 1)  # 一様分布
                        iter += 1

            return network, iter
        else:
            # EfficientNet の場合
            num_ftrs = network.classifier.in_features
            fc = nn.Linear(num_ftrs, out_features=num_classes)
            network.classifier = fc
            return network, 1
    # ResNet の場合
    elif hasattr(network, "fc"):
        num_ftrs = network.fc.in_features
        fc = nn.Linear(num_ftrs, out_features=num_classes)
        network.fc = fc
        return network, 1
    # IncResNetV2 の場合
    elif hasattr(network, "classif"):
        num_ftrs = network.classif.in_features
        fc = nn.Linear(num_ftrs, out_features=num_classes)
        network.classif = fc
        return network, 1


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
