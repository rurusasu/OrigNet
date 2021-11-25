import os
import sys
from typing import Dict, List, Literal, Union

sys.path.append("../../../")

import cv2
import numpy as np
import skimage.io as io
import torch.utils.data as data
from torchvision import transforms
from yacs.config import CfgNode

from lib.config.config import pth
from lib.datasets.ARCdataset.ARCutils import (
    FilterARCDataset,
    getARCBinaryMask,
    getARCNormalMask,
)
from lib.datasets.COCOdataset.COCOutils import (
    FilterCOCODataset,
    getCOCOBinaryMask,
    getCOCONormalMask,
)


def getImage(imgObj, img_folder: str, input_img_size: Dict) -> np.ndarray:
    # Read and normalize an image
    # ndarray([H, W, C])
    img = io.imread(os.path.join(img_folder, imgObj["file_name"])) / 255.0
    # Resize: [H, W, C] -> [H', W', C]
    # 変数が，[W, H] で与える点に注意
    img = cv2.resize(img, (input_img_size["w"], input_img_size["h"]))
    if len(img.shape) == 3 and img.shape[2] == 3:  # If it is a RGB 3 channel image
        return img
    else:  # 白黒の画像を扱う場合は、次元を3にする
        stacked_img = np.stack((img,) * 3, axis=-1)
        return stacked_img


class SegmentationDataset(data.Dataset):
    def __init__(
        self,
        data_root: str,
        cls_names: Union[List[str], None] = None,
        split: Literal["train", "val", "test"] = "train",
        img_shape: Dict[str, int] = {"width": 224, "height": 224},
        mask_type: Literal["binary", "normal"] = "normal",
        transforms: Union[transforms.Compose, None] = None,
    ):
        """セマンティックセグメンテーションのタスクで使用するデータセットを読みだすクラスの初期化関数．

        Args:
            data_root (str): 親ディレクトリのパス．
            cls_names (Union[List[str], None], optional):
                読みだしたいクラス名のリスト.
                `None` の場合，すべてのクラスを読みだす．
                Defaults to None.
            split (Literal[, optional):
                どのデータセットを読みだすか.
                ["train", "val", "test"] の3つから選択可能．
                Defaults to "train".
            img_shape (Dict[str, int], optionanl):
                出力される画像のサイズ．
                Default to {"width": 224, "height": 224}.
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
        self.cls_names = cls_names
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        self.img_shape = img_shape
        self.mask_type = mask_type
        self.split = split
        self.transforms = transforms

        self.img_dir = os.path.join(self.data_root, self.split, "rgb")
        self.ann_dir = os.path.join(self.data_root, self.split)

        if not os.path.exists(self.img_dir) or not os.path.isdir(self.img_dir):
            raise FileExistsError(
                f"The dataset to be used for {self.img_dir} could not be read. The path is invalid."
            )

        if "COCO" in data_root:
            (
                self.imgs_info,
                self.dataset_size,
                self.coco,
                self.cls_names,
            ) = FilterCOCODataset(self.data_root, self.cls_names, self.split)
        else:
            (
                self.imgs_info,
                self.dataset_size,
                self.coco,
                self.cls_names,
            ) = FilterARCDataset(self.data_root, self.cls_names, self.split)
        self.catIds = self.coco.getCatIds(catNms=self.cls_names)

    def __getitem__(self, img_id):
        if type(img_id) is tuple:
            img_id, height, width = img_id
        elif type(img_id) is int:
            width, height = self.img_shape["width"], self.img_shape["height"]
        else:
            raise TypeError("Invalid type for variable index")

        img_info = self.imgs_info[img_id]
        del img_id

        input_img_size = {}
        input_img_size["w"] = width
        input_img_size["h"] = height

        if "COCO" in self.data_root:
            ### Retrieve Image ###
            img = getImage(
                imgObj=img_info, img_folder=self.img_dir, input_img_size=input_img_size
            )
            ### Create Mask ###
            if self.mask_type == "binary":
                mask = getCOCOBinaryMask(
                    img_info, self.coco, self.catIds, input_img_size
                )
            elif self.mask_type == "normal":
                mask, class_names = getCOCONormalMask(
                    img_info, self.cls_names, self.coco, self.catIds, input_img_size
                )
        else:
            ### Retrieve Image ###
            img = getImage(
                imgObj=img_info, img_folder=self.img_dir, input_img_size=input_img_size
            )
            ### Create Mask ###
            if self.mask_type == "binary":
                mask = getARCBinaryMask(
                    img_info, self.coco, self.catIds, input_img_size
                )
            elif self.mask_type == "normal":
                mask = getARCNormalMask(
                    img_info,
                    self.cls_names,
                    self.coco,
                    self.catIds,
                    self.ann_dir,
                    input_img_size,
                )

        # del img_info, input_img_size
        # gc.collect()

        if self.transforms is not None:
            img, mask = self.transforms.augment(img=img, mask=mask, split=self.split)

        # ndarray -> tensor
        # img = torch.from_numpy(img.astype(np.float32)).clone()
        # mask = torch.from_numpy(mask.astype(np.float32)).clone()

        ret = {
            "img": img,
            "target": mask,
            "meta": self.split,
            "cls_names": self.cls_names,
        }
        return ret

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from lib.datasets.make_datasets import make_data_loader
    from lib.visualizers.segmentation import visualize

    conf = CfgNode()
    conf.task = "semantic_segm"
    # conf.cls_names = ["laptop", "tv"]
    conf.cls_names = ["item18", "item39"]
    # conf.cls_names = None
    conf.img_width = 400
    conf.img_height = 400
    conf.train = CfgNode()
    # conf.train.dataset = "COCO2017Val"
    conf.train.dataset = "ARCTrain"
    conf.train.batch_size = 4
    conf.train.num_workers = 1
    conf.train.batch_sampler = ""
    conf.test = CfgNode()
    # conf.test.dataset = "COCO2017Val"
    conf.test.dataset = "ARCTest"
    conf.test.batch_size = 4
    conf.test.num_workers = 1
    conf.test.batch_sampler = ""

    dloader = make_data_loader(conf, ds_category="train", is_distributed=True)
    batch_iter = iter(dloader)
    batch = next(batch_iter)
    img, mask = batch["img"], batch["target"]
    img, mask = img[1, :, :, :], mask[1, :, :]
    fig = visualize(input=img, mask=mask)
    plt.show()
