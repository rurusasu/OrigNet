import sys
from typing import Literal

sys.path.append("..")
sys.path.append("../../")


import timm

# timm の github
# REF: https://github.com/rwightman/gen-efficientnet-pytorch
# 関数化
# REF:https://dajiro.com/entry/2020/07/24/161040


def ResNet_18(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("resnet18", pretrained=pretrained, **kwargs)
    return model


def ResNet_34(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("resnet34", pretrained=pretrained, **kwargs)
    return model


def ResNet_50(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("resnet50", pretrained=pretrained, **kwargs)
    return model


def ResNet_101(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("resnet101", pretrained=pretrained, **kwargs)
    return model


def ResNet_152(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("resnet152", pretrained=pretrained, **kwargs)
    return model


ResNet_spec = {
    "18": ResNet_18,
    "34": ResNet_34,
    "50": ResNet_50,
    "101": ResNet_101,
    "152": ResNet_152,
}


def get_res_net(
    model_num: Literal[18, 34, 50, 101, 152], pretrained: bool = False, **kwargs
):
    model = ResNet_spec[model_num](pretrained, **kwargs)

    return model
