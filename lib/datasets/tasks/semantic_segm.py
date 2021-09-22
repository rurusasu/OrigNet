import os
import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")

from typing import Type, Union, Dict, List

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from yacs.config import CfgNode

from datasets.augmentation import augmentation
from lib.config.config import pth
from lib.utils.base_utils import GetImgFpsAndLabels, LoadImgs


class Dataset(data.Dataset):
    """data_root の子ディレクトリ名がクラスラベルという仮定のもとデータセットを作成するクラス．
    データセットは以下のような構成を仮定
    dataset_root
          |
          |- train
          |     |
          |     |- ObjectName_1
          |     |           |
          |     |           |- images
          |     |           |
          |     |           |- masks
          |     |
          |     |- ObjectName_2
          |     |
          |
          |- test
    """

    def __init__(
        self,
        cfg: CfgNode,
        data_root: str,
        split: str,
        cls_names: List[str] = None,
        transforms: transforms = None,
    ) -> None:
        super(Dataset, self).__init__()

        self.cfg = cfg
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        (
            self.classes,
            self.class_to_idx,
            self.imgs,
            self.targets,
            self.msks,
        ) = GetImgFpsAndLabels(self.data_root)
        self.split = split
        self._transforms = transforms

        # 入力されたクラス名が None 以外でかつ取得したクラスラベルに含まれている場合
        if cls_names is not None and cls_names in self.classes:
            self.cls_names = cls_names
        else:
            self.cls_names = None

    def __getitem__(self, img_id: Type[Union[int, tuple]]) -> Dict:
        """
        データセット中から `img_id` で指定された番号のデータを返す関数．

        Arg:
            img_id (Type[Union[int, tuple]]): 読みだすデータの番号

        Return:
            ret (Dict["img": torch.tensor,
                         "msk": torch.tensor,
                         "meta": str,
                         "target": int,
                         "cls_name": str]):
        """
        if type(img_id) is tuple:
            img_id, height, width = img_id
        elif (
            type(img_id) is int and "img_width" in self.cfg and "img_height" in self.cfg
        ):
            height, width = self.cfg.img_width, self.cfg.img_height
        else:
            raise TypeError("Invalid type for variable index")

        # images (rgb, mask) の読み出し
        imgs = LoadImgs(self.imgs, img_id, self.msks)

        # `OpenCV` および `numpy` を用いたデータ拡張
        if self.split == "train":
            imgs = augmentation(imgs, height, width, self.split)

        # 画像をテンソルに変換
        img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((width, height))]
        )
        for k in imgs.keys():
            if len(imgs[k]) > 0:
                imgs[k] = img_transforms(
                    Image.fromarray(np.ascontiguousarray(imgs[k], np.uint8))
                )

        # `transforms`を用いた変換がある場合は行う．
        if self._transforms is not None:
            imgs["img"] = self._transforms(imgs["img"])

        ret = {
            "img": imgs["img"],
            "msk": imgs["msk"],
            "meta": self.split,
            "target": self.targets[img_id],
            "cls_name": self.classes[self.targets[img_id]],
        }
        return ret

    def __len__(self):
        """ディレクトリ内の画像ファイル数を返す関数．"""
        return len(self.imgs)


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_dataset

    cfg = CN()
    cfg.task = "semantic_segm"
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CN()
    cfg.train.dataset = "LinemodTrain"

    dataset = make_dataset(cfg, cfg.train.dataset)
    print(dataset)
