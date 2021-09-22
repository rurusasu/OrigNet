import os
import sys
from glob import glob

sys.path.append("../../")

from typing import Type, Union

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from yacs.config import CfgNode

from datasets.augmentation import augmentation
from lib.config.config import pth


class Dataset(data.Dataset):
    """次のようなデータセットの構成を仮定
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
        self, cfg: CfgNode, data_root: str, split: str, transforms: transforms = None
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
        ) = self._get_img_pths_labels(self.data_root)
        self.split = split
        self._transforms = transforms

    def __getitem__(self, img_id: Type[Union[int, tuple]]) -> tuple:
        """data_root の子ディレクトリ名が教師ラベルと仮定"""
        if type(img_id) is tuple:
            img_id, height, width = img_id
        elif (
            type(img_id) is int and "img_width" in self.cfg and "img_height" in self.cfg
        ):
            height, width = self.cfg.img_width, self.cfg.img_height
        else:
            raise TypeError("Invalid type for variable index")

        # img, cls_num = self._read_img(self.imgs, img_id)
        img = self._read_img(self.imgs, img_id)

        # データ拡張の処理を記述
        if self.split == "train":
            img = augmentation(img, height, width, self.split)
        else:
            img = img

        # 画像をテンソルに変換
        img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = img_transforms(Image.fromarray(np.ascontiguousarray(img, np.uint8)))

        # テンソルを用いた変換がある場合は行う．
        if self._transforms is not None:
            img = self._transforms(img)

        ret = {
            "img": img,
            "msk": [],
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
                             例: [(img_fp1, cls_num1), (img_fp2, cls_num1), ...]
            targets (list): cls_num を格納したリスト
        """
        # train の子ディレクトリ名を教師ラベルとして設定
        classes = []
        class_to_idx = {}
        imgs = []
        targets = []
        for i, p in enumerate(glob(os.path.join(data_root, "*"))):
            cls_name = os.path.basename(p.rstrip(os.sep))
            # クラス名のリストを作成
            classes.append(cls_name)
            # クラス名と label_num を対応させる辞書を作成
            class_to_idx[cls_name] = i

            # クラス名ディレクトリ内の file_ext にヒットするパスを全探索
            # RGB 画像を探索
            for _, img_pth in enumerate(glob(os.path.join(p, "**"), recursive=True)):
                if (
                    os.path.isfile(img_pth)
                    and os.path.splitext(img_pth)[1] in self.file_ext
                ):
                    # データパスと label_num を格納したタプルを作成
                    # imgs.append((img_pth, i))
                    imgs.append(img_pth)
                    # label_num のみ格納したリストを作成
                    targets.append(i)

        return classes, class_to_idx, imgs, targets

    def _read_img(self, imgs: list, img_id: int):
        """画像パスのリストから、id で指定された画像を読みだす関数

        Args:
            igm_pths (list): 画像パスのリスト
            img_id (int): 読みだすパスの番号

        Return:

        """
        # img_pth, cls_num = imgs[img_id][0], imgs[img_id][1]
        img_pth = imgs[img_id]
        # 画像パスが存在する場合
        # if os.path.exists(img_pth):
        try:
            # 画像を読み込む
            img = Image.open(img_pth)
        except Exception as e:
            print(f"Read image error: {e}")
        else:
            # return img, cls_num
            return img
