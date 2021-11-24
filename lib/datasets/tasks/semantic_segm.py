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
        cfg: CfgNode,
        data_root: str,
        cls_names: Union[List[str], None] = None,
        # input_img_size: tuple = (224, 224),
        split: Literal["train", "val", "test"] = "train",
        mask_type: Literal["binary", "normal"] = "normal",
        transforms: Union[transforms.Compose, None] = None,
    ):
        """セマンティックセグメンテーションのタスクで使用するデータセットを読みだすクラスの初期化関数．

        Args:
            cfg (CfgNode): 訓練の条件設定が保存された辞書．
            data_root (str): 親ディレクトリのパス．
            cls_names (Union[List[str], None], optional): 読みだしたいクラス名のリスト.
            `None` の場合，すべてのクラスを読みだす．Defaults to None.
            split (Literal[, optional): どのデータセットを読みだすか.
            ["train", "val", "test"] の3つから選択可能．Defaults to "train".
            mask_type (Literal[, optional):
            "binary": すべてのオブジェクトを単一のクラスとしてマスクする．
            "normal": オブジェクトをクラスごとにマスクする
            Defaults to "normal".
            transforms (Union[transforms.Compose, None], optional): データ拡張で使用するクラス．
            Noneの場合は，データ拡張を行わない．Defaults to None.

        Raises:
            FileExistsError: [description]
        """
        self.cfg = cfg
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        self.cls_names = cfg.cls_names
        self.split = split
        self.mask_type = mask_type

        self.img_dir = os.path.join(self.data_root, self.split, "rgb")
        self.ann_dir = os.path.join(self.data_root, self.split)

        if not os.path.exists(self.img_dir) or not os.path.isdir(self.img_dir):
            raise FileExistsError(
                f"The dataset to be used for {cfg.task} could not be read. The path is invalid."
            )

        # imgs_info = {
        # license: int,
        # file_name: str, 例: 000000495776.jpg
        # coco_url: str, 例: http://images.cocodataset.org/train2017/000000495776.jpg
        # height: int, 例 375
        # width: int, 例 500
        # date_captured, 例 2013-11-24 07:55:36
        # flickr_url: str, 例 http://farm1.staticflickr.com/21/30368166_92245cce3f_z.jpg
        # id: int 例 495776
        # }
        if "COCO" in self.cfg.train.dataset or "COCO" in self.cfg.test.dataset:
            self.imgs_info, self.dataset_size, self.coco = FilterCOCODataset(
                self.data_root, self.cls_names, self.split
            )
        else:
            self.imgs_info, self.dataset_size, self.coco = FilterARCDataset(
                self.data_root, self.cls_names, self.split
            )
        self.catIds = self.coco.getCatIds(catNms=self.cls_names)

        # Data Augmentation
        self.transforms = transforms

    def __getitem__(self, img_id):
        if type(img_id) is tuple:
            img_id, height, width = img_id
        elif (
            type(img_id) is int and "img_width" in self.cfg and "img_height" in self.cfg
        ):
            width, height = self.cfg.img_width, self.cfg.img_height
        else:
            raise TypeError("Invalid type for variable index")

        img_info = self.imgs_info[img_id]
        del img_id

        input_img_size = {}
        input_img_size["w"] = width
        input_img_size["h"] = height

        if "COCO" in self.cfg.train.dataset or "COCO" in self.cfg.test.dataset:
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
