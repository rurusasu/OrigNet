import os
import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")

from typing import Dict, List, Literal, Tuple, Type, Union

import torch
import torch.utils.data as data
from torchvision import transforms
from yacs.config import CfgNode

from lib.config.config import pth
from lib.utils.base_utils import GetImgFpsAndLabels, LoadImgAndResize


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
        data_root: str,
        cls_names: Union[List[str], None] = None,
        split: Literal["train", "val", "test"] = "train",
        img_shape: Dict[str, int] = {"width": 224, "height": 224},
        mask_type: Literal["binary", "normal"] = "normal",
        transforms: Union[transforms.Compose, None] = None,
    ) -> None:
        """分類タスクで使用するデータセットを読み出すクラスの初期化関数．

        Args:
            data_root (str): 親ディレクトリのパス．
            cls_names (Union[List[str], None], optional):
                読みだしたいクラス名のリスト. 現在は使用されていないダミー変数．
                Defaults to None.
            split (Literal[, optional):
                どのデータセットを読みだすか.
                ["train", "val", "test"] の3つから選択可能．
                Defaults to "train".
            img_shape (Dict[str, int], optional):
                出力される画像のサイズ.
                Defaults to {"width": 224, "height": 224}.
            mask_type (Literal["binary", "normal"], optional):
                "binary": すべてのオブジェクトを単一のクラスとしてマスクする．
                "normal": オブジェクトをクラスごとにマスクする.
                Defaults to "normal".
            transforms (Union[transforms.Compose, None], optional):
                データ拡張で使用するクラス．
                Noneの場合は，データ拡張を行わない．
                Defaults to None.

        Raises:
            FileExistsError: [description]
        """

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
        self.cls_names = cls_names
        self.img_shape = img_shape
        self.transforms = transforms
        self.split = split
        self.img_dir = os.path.join(pth.DATA_DIR, data_root, self.split)

        if not os.path.exists(self.img_dir) or not os.path.isdir(self.img_dir):
            raise FileExistsError(
                f"The dataset to be used for {self.img_dir} could not be read. The path is invalid."
            )

        (
            self.cls_names,
            self.class_to_idx,
            self.img_fps,
            self.targets,
            _,
        ) = GetImgFpsAndLabels(self.img_dir, cls_names=self.cls_names)

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
        elif type(img_id) is int:
            # height, width = self.cfg.img_width, self.cfg.img_height
            width, height = self.img_shape["width"], self.img_shape["height"]
        else:
            raise TypeError("Invalid type for variable index")

        input_img_size = {}
        input_img_size["w"] = width
        input_img_size["h"] = height
        # 画像を読みだす
        img_fp = self.img_fps[img_id]
        img = LoadImgAndResize(img_fp, input_img_size=input_img_size)

        # `transforms`を用いた変換がある場合は行う．
        if self.transforms is not None:
            img, _ = self.transforms.augment(img=img)

        ret = {
            "img": img,
            "target": torch.tensor(self.targets[img_id]),
            "meta": self.split,
            "cls_name": self.cls_names[self.targets[img_id]],
        }
        return ret

    def __len__(self):
        """ディレクトリ内の画像ファイル数を返す関数．"""
        return len(self.img_fps)


if __name__ == "__main__":
    from lib.datasets.make_datasets import make_data_loader
    from lib.visualizers.segmentation import visualize

    cfg = CfgNode()
    cfg.task = "classify"
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CfgNode()
    cfg.train.dataset = "SampleTrain"
    cfg.train.batch_size = 4
    cfg.train.num_workers = 2
    cfg.train.batch_sampler = ""

    dloader = make_data_loader(cfg, is_train=True)
    batch_iter = iter(dloader)
    batch = next(batch_iter)
    img, _ = batch["img"], batch["target"]
    img = img[1, :, :, :]
    visualize(imgs=img)
