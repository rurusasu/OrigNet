import sys
from typing import Literal

sys.path.append("..")
sys.path.append("../../")

import timm


# timm の github
# REF: https://github.com/rwightman/gen-efficientnet-pytorch
# 関数化
# REF:https://dajiro.com/entry/2020/07/24/161040


def EfficientNet_b0(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, **kwargs)
    return model


EfficientNet_spec = {"b0": EfficientNet_b0}


def get_efficient_net(
    model_num: Literal["b0", "b2"], pretrained: bool = False, **kwargs
):
    model = EfficientNet_spec[model_num](pretrained, **kwargs)

    return model
