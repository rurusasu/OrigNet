import os
import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")

from typing import Type, Union, Dict, List

import numpy as np
import torch.utils.data as data
from torchvision import transforms
from yacs.config import CfgNode

from datasets.augmentation import augmentation
from lib.config.config import pth
from lib.utils.base_utils import GetImgFpsAndLabels, LoadImgs


class ClassifyDataset(data.Dataset):
    """data_root の子ディレクトリ名がクラスラベルという仮定のもとデータセットを作成するクラス．
    データセットは以下のような構成を仮定
    dataset_root
          |
          |- train
          |     |
          |     |- OK
          |     |
          |     |- NG
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
        super(ClassifyDataset, self).__init__()

        self.file_ext = {
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        }
        self.cfg = cfg
        self.split = split
        self.img_dir = os.path.join(pth.DATA_DIR, data_root, self.split)
        (
            self.cls_names,
            self.class_to_idx,
            self.img_fps,
            self.targets,
            _,
        ) = GetImgFpsAndLabels(self.img_dir)
        self._transforms = transforms

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
        imgs = LoadImgs(self.img_fps, img_id)

        if len(imgs["img"].shape) == 2:
            imgs["img"] = np.stack([imgs["img"], imgs["img"], imgs["img"]], axis=2)
        elif len(imgs["img"].shape) == 3:
            pass
        else:
            raise ValueError("Incorrect way of giving image size.")

        # `OpenCV` および `numpy` を用いたデータ拡張
        if self.split == "train":
            imgs = augmentation(imgs, height, width, self.split)

        # `transforms`を用いた変換がある場合は行う．
        if self._transforms is not None:
            imgs["img"] = self._transforms(imgs["img"])

        ret = {
            "img": imgs["img"],
            "target": self.targets[img_id],
            "meta": self.split,
            "cls_names": self.cls_names[self.targets[img_id]],
        }
        return ret

    def __len__(self):
        """ディレクトリ内の画像ファイル数を返す関数．"""
        return len(self.img_fps)


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_dataset

    cfg = CN()
    cfg.task = "classify"
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CN()
    cfg.train.dataset = "SampleTrain"

    dataset = make_dataset(cfg, cfg.train.dataset)
    print(dataset)
