import os
import sys
import random
from typing import Dict, List, Literal, Union

sys.path.append("../../../")

import cv2
import numpy as np
import skimage.io as io
import torch.utils.data as data
from pycocotools.coco import COCO
from torchvision import transforms
from yacs.config import CfgNode

from lib.config.config import pth


def getClassName(classID: int, cats: dict):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


def FilterDataset(
    data_root,
    cls_names: Union[List[str], None] = None,
    split: Literal["train", "val", "test"] = "train",
):
    """フィルタしたクラスのオブジェクトが映る画像をすべて読みだす関数

    Args:
        data_root (str): データセットの root ディレクトリ．
        cls_names (Union(List[str], None), optional): 抽出するクラス名のリスト. Defaults to None.
        split (Literal["train", "val", "test"], optional): 読みだすデータセットの種類（'train' or 'val', or 'test'）. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    # initialize COCO api for instance annotations
    annFile = "{}/annotations/instances_{}2017.json".format(data_root, split)
    coco = COCO(annFile)

    images = []
    if cls_names is not None:
        # リスト内の個々のクラスに対してイテレートする
        for className in cls_names:
            # 与えられたカテゴリを含むすべての画像を取得する
            catIds = coco.getCatIds(catNms=className)  # <- ann
            imgIds = coco.getImgIds(catIds=catIds)  # <- ann
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # del annFile, catIds, imgIds
    # gc.collect()

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])
    # del images
    # gc.collect()

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


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


def getNormalMask(imgObj, cls_names, coco, catIds, input_img_size):
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # del annIds
    # gc.collect()

    cats = coco.loadCats(catIds)
    mask = np.zeros((input_img_size["h"], input_img_size["w"]))  # mask [H, W]
    class_names = []
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = cls_names.index(className) + 1
        # ndarray [H, W] -> ndarray[H', W']
        new_mask = cv2.resize(
            coco.annToMask(anns[a]) * pixel_value,
            (input_img_size["w"], input_img_size["h"]),
        )
        mask = np.maximum(new_mask, mask)
        class_names.append(className)

    # del anns, cats
    # gc.collect()
    # Add extra dimension for parity with train_img size [X * X * 3]
    # mask = mask.reshape(input_img_size[0], input_img_size[1], 1)

    # すべての
    return mask, class_names


def getBinaryMask(imgObj, coco, catIds, input_img_size) -> np.ndarray:
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)  # アノテーションを読みだす
    # del annIds
    # gc.collect()

    # train_mask = np.zeros(input_img_size)
    mask = np.zeros(input_img_size)
    for id in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[id]), input_img_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        # 画素の位置ごとの最大値を返す
        mask = np.maximum(new_mask, mask)

    # del anns, new_mask
    # gc.collect()
    # パリティ用の追加次元をtrain_imgのサイズ[X * X * 3]で追加。
    mask = mask.reshape(input_img_size[0], input_img_size[1], 1)
    return mask


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
        self.cfg = cfg
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        self.cls_names = cfg.cls_names
        self.split = split
        self.mask_type = mask_type

        self.img_dir = os.path.join(self.data_root, self.split)

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
        self.imgs_info, self.dataset_size, self.coco = FilterDataset(
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
        ### Retrieve Image ###
        img = getImage(
            imgObj=img_info, img_folder=self.img_dir, input_img_size=input_img_size
        )

        class_names = ["object"]
        ### Create Mask ###
        if self.mask_type == "binary":
            mask = getBinaryMask(img_info, self.coco, self.catIds, input_img_size)

        elif self.mask_type == "normal":
            mask, class_names = getNormalMask(
                img_info, self.cls_names, self.coco, self.catIds, input_img_size
            )

        # del img_info, input_img_size
        # gc.collect()

        if self.transforms is not None:
            img, mask = self.transforms.augment(img=img, mask=mask)
            # img = self.transforms(img)
            # mask = self.transforms(mask)

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
    from lib.datasets.make_datasets import make_data_loader
    from lib.visualizers.segmentation import visualize

    cfg = CfgNode()
    cfg.task = "semantic_segm"
    cfg.cls_names = ["laptop", "tv"]
    cfg.img_width = 600
    cfg.img_height = 200
    cfg.train = CfgNode()
    cfg.train.dataset = "COCO2017Val"
    cfg.train.batch_size = 4
    cfg.train.num_workers = 2
    cfg.train.batch_sampler = ""
    cfg.test = CfgNode()
    cfg.test.dataset = "COCO2017Val"
    cfg.test.batch_size = 4
    cfg.test.num_workers = 2
    cfg.test.batch_sampler = ""

    dloader = make_data_loader(cfg, is_train=True)
    batch_iter = iter(dloader)
    batch = next(batch_iter)
    img, mask = batch["img"], batch["target"]
    img, mask = img[1, :, :, :], mask[1, :, :]
    visualize(imgs=img, msk=mask)
