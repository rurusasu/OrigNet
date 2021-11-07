import sys
from typing import Literal

sys.path.append("..")
sys.path.append("../../")

import timm


# timm の github
# REF: https://github.com/rwightman/pytorch-image-models
# timm の Docs
# REF: https://fastai.github.io/timmdocs/
# Pytorch画像モデル
# REF: https://rwightman.github.io/pytorch-image-models/
# 関数化
# REF:https://dajiro.com/entry/2020/07/24/161040


def InceptionV3(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("inception_v3", pretrained=pretrained, **kwargs)
    return model


def InceptionV4(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("inception_v4", pretrained=pretrained, **kwargs)
    return model


def InceptionResNetV2(pretrained: bool = False, **kwargs):
    # モデルの定義
    model = timm.create_model("inception_resnet_v2", pretrained=pretrained, **kwargs)
    return model


Inception_spec = {"v3": InceptionV3, "v4": InceptionV4, "res_v2": InceptionResNetV2}


def get_inception_net(
    model_num: Literal["v3", "v4", "res_v2"], pretrained: bool = False, **kwargs
):
    model = Inception_spec[model_num](pretrained, **kwargs)

    return model
