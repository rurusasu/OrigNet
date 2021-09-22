import os
import sys
from glob import glob

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")

from typing import Type, Union, List, Dict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from yacs.config import CfgNode

from datasets.augmentation import augmentation
from lib.config.config import pth
from lib.utils.base_utils import load_img


class Dataset(data.Dataset):
    """次のようなデータセットの構成を仮定
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
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        # self.img_pths = self._get_img_pths_labels(self.data_root)
        (
            self.classes,
            self.class_to_idx,
            self.imgs,
            self.targets,
            self.msks,
        ) = self._get_img_pths_labels(self.data_root)
        self.split = split
        self._transforms = transforms

        # 入力されたクラス名が None 以外でかつ取得したクラスラベルに含まれている場合
        if cls_names is not None and cls_names in self.classes:
            self.cls_names = cls_names
        else:
            self.cls_names = None

    def __getitem__(self, img_id: Type[Union[int, tuple]]) -> Dict:
        """
        data_root の子ディレクトリ名が教師ラベルと仮定

        Arg:
            img_id (Type[Union[int, tuple]]): 読みだす画像の番号

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
        imgs = load_img(self.imgs, img_id, self.msks)

        # データ拡張の処理を記述
        if self.split == "train":
            imgs = augmentation(imgs, height, width, self.split)

        # 画像をテンソルに変換
        img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((width, height))]
        )
        for k in imgs.keys():
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

    def _get_img_pths_labels(self, data_root: str):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する．

        Arg:
            data_root (str): 画像データが格納された親フォルダ
        Return:
            classes (list): クラス名のリスト
            class_to_idx (dict): クラス名と label_num を対応させる辞書を作成
            imgs (list): データパスと label_num を格納したタプルを作成．
                             例: [img_fp1, img_fp2, ...]
            targets (list): cls_num を格納したリスト
        """
        # train の子ディレクトリ名を教師ラベルとして設定
        classes = []
        class_to_idx = {}
        imgs = []
        msks = []
        targets = []
        for i, p in enumerate(glob(os.path.join(data_root, "*"))):
            cls_name = os.path.basename(p.rstrip(os.sep))
            # クラス名のリストを作成
            classes.append(cls_name)
            # クラス名と label_num を対応させる辞書を作成
            class_to_idx[cls_name] = i

            # クラス名ディレクトリ内の file_ext にヒットするパスを全探索
            # RGB 画像を探索
            for img_pth in glob(os.path.join(p, "imgs", "**"), recursive=True):
                if (
                    os.path.isfile(img_pth)
                    and os.path.splitext(img_pth)[1] in self.file_ext
                ):
                    # 画像データパスをリストに格納
                    imgs.append(img_pth)
                    # label_num をリストに格納
                    targets.append(i)

            for msk_pth in glob(os.path.join(p, "masks", "**"), recursive=True):
                if (
                    os.path.isfile(msk_pth)
                    and os.path.splitext(msk_pth)[1] in self.file_ext
                ):
                    # マスク画像データパスをリストに格納
                    msks.append(msk_pth)

        return classes, class_to_idx, imgs, targets, msks


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
