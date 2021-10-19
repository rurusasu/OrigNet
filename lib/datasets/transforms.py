from typing import Tuple, Union

import albumentations as albu
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Normalize

from yacs.config import CfgNode


def to_tensor(x, **kwargs):
    # ndarray([H, W, C]) -> torch.tensor([C, H, W])
    x = torch.from_numpy(x.astype(np.float32)).clone()
    x = x.permute(2, 0, 1)
    return x


def mask_to_tensor(x, **kwargs):
    # ndarray([H, W]) -> torch.tensor([H, W])
    x = torch.from_numpy(x.astype(np.float32)).clone()
    return x


class DataAugmentor(object):
    # ImageNet の画素値の平均値と分散
    # REF: https://pytorch.org/vision/stable/models.html
    image_net = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def __init__(self, cfg: CfgNode, is_train: bool) -> None:
        self.is_train = is_train if is_train else False

        self.transform = {}
        self.transform["train"] = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.Lambda(image=to_tensor, mask=mask_to_tensor),
            ]
        )
        self.transform["val"] = albu.Compose(
            [albu.Lambda(image=to_tensor, mask=mask_to_tensor)]
        )
        self.transform["norm"] = transforms.Compose(
            [Normalize(mean=self.image_net["mean"], std=self.image_net["std"])]
        )

    def augment(
        self,
        img: np.ndarray,
        mask: Union[np.ndarray, None] = None,
        is_train: Union[bool, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        画像オーギュメンテーションを訓練，検証で施す処理を変える関数．

        Args:
            img(np.ndarray): サイズが `[H, W, C]` の ndarray 配列．
            mask(np.ndarray): サイズが `[H, W]` の ndarray 配列．
        """
        is_train = is_train if is_train is not None else self.is_train
        # データオーギュメンテーションを実行する．
        # マスク画像が存在する場合
        if mask is not None:
            if is_train:  # 訓練用
                sample = self.transform["train"](image=img, mask=mask)
            else:  # 検証・テスト用
                sample = self.transform["val"](image=img, mask=mask)
            img, mask = sample["image"], sample["mask"]
        else:  # マスク画像が存在しない場合
            if is_train:  # 訓練用
                sample = self.transform["train"](image=img)
            else:  # 検証・テスト用
                sample = self.transform["val"](image=img)
            img, mask = sample["image"], []

        # 入力画像のみ標準化する．
        img = self.transform["norm"](img)

        return img, mask


def make_transforms(cfg: CfgNode, is_train: bool) -> object:
    """データ拡張に使用する Transforms を作成する関数
    Args:
        cfg (CfgNode): データセット名などのコンフィグ情報
        is_train (bool): 訓練用データセットか否か．

    Return:
        (object): データ変換に使用する関数群
    """
    return DataAugmentor(cfg, is_train=is_train)
