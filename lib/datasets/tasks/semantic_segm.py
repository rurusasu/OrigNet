import os
import sys
from glob import glob

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")

from typing import Tuple, Type, Union, List

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

    def __getitem__(self, img_id: Type[Union[int, tuple]]) -> tuple:
        """
        data_root の子ディレクトリ名が教師ラベルと仮定

        Arg:
           img_id (Type[Union[int, tuple]]): 読みだす画像の番号

        Return:
            (tuple):
        """
        if type(img_id) is tuple:
            img_id, height, width = img_id
        elif (
            type(img_id) is int and "img_width" in self.cfg and "img_height" in self.cfg
        ):
            height, width = self.cfg.img_width, self.cfg.img_height
        else:
            raise TypeError("Invalid type for variable index")

        (
            img,
            msk,
        ) = self._read_img(self.imgs, self.msks, img_id)

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
            "msk": msk,
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
                             例: [(img_fp1), (img_fp2), ...]
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
            for img_pth in glob(os.path.join(p, "imgs", "**"), recursive=True):
                if (
                    os.path.isfile(img_pth)
                    and os.path.splitext(img_pth)[1] in self.file_ext
                ):
                    # データパスと label_num を格納したタプルを作成
                    imgs.append(img_pth)
                    # label_num のみ格納したリストを作成
                    targets.append(i)
            masks = [
                msk_pth
                for msk_pth in glob(os.path.join(p, "masks", "**"), recursive=True)
                if (
                    os.path.isfile(msk_pth)
                    and os.path.splitext(msk_pth)[1] in self.file_ext
                )
            ]

        return classes, class_to_idx, imgs, targets, masks

    def _read_img(
        self,
        imgs: List[Tuple[str]],
        msks: List[str],
        img_id: int,
    ) -> Image:
        """画像パスのリストから、id で指定された画像を読みだす関数

        Args:
            igm_pths (list): 画像パスのリスト
            img_id (int): 読みだすパスの番号

        Return:

        """
        img_pth = imgs[img_id]
        msk_pth = msks[img_id]
        # 画像パスが存在する場合
        # if os.path.exists(img_pth):
        try:
            # 画像を読み込む
            img = Image.open(img_pth)
            msk = Image.open(msk_pth)
        except Exception as e:
            print(f"Read image error: {e}")
        else:
            return img, msk


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
