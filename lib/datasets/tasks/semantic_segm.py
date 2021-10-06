import os
import sys
import random
from typing import List, Literal, Union

sys.path.append("../../../")

import cv2
import numpy as np
import skimage.io as io
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, dataloader
from torchvision.transforms import Compose
from yacs.config import CfgNode

from lib.config.config import pth

### For visualizing the outputs ###
import matplotlib.pyplot as plt


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
    if cls_names != None:
        # リスト内の個々のクラスに対してイテレートする
        for className in cls_names:
            # 与えられたカテゴリを含むすべての画像を取得する
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


def getImage(imgObj, img_folder: str, input_img_size: tuple) -> np.ndarray:
    # Read and normalize an image
    img = io.imread(os.path.join(img_folder, imgObj["file_name"])) / 255.0
    # Resize
    img = cv2.resize(img, input_img_size)
    if len(img.shape) == 3 and img.shape[2] == 3:  # If it is a RGB 3 channel image
        return img
    else:  # 白黒の画像を扱う場合は、次元を3にする
        stacked_img = np.stack((img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imgObj, cls_names, coco, catIds, input_img_size):
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    mask = np.zeros(input_img_size)
    class_names = []
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = cls_names.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_img_size)
        mask = np.maximum(new_mask, mask)
        class_names.append(className)

    # Add extra dimension for parity with train_img size [X * X * 3]
    mask = mask.reshape(input_img_size[0], input_img_size[1], 1)
    return mask, class_names


def getBinaryMask(imgObj, coco, catIds, input_img_size) -> np.ndarray:
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)  # アノテーションを読みだす
    # train_mask = np.zeros(input_img_size)
    mask = np.zeros(input_img_size)
    for id in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[id]), input_img_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        # 画素の位置ごとの最大値を返す
        mask = np.maximum(new_mask, mask)

    # パリティ用の追加次元をtrain_imgのサイズ[X * X * 3]で追加。
    mask = mask.reshape(input_img_size[0], input_img_size[1], 1)
    return mask


class SegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        data_root: str,
        cls_names: Union[List[str], None] = None,
        # input_img_size: tuple = (224, 224),
        split: Literal["train", "val", "test"] = "train",
        mask_type: Literal["binary", "normal"] = "normal",
        transforms: Union[Compose, None] = None,
    ):
        self.cfg = cfg
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        self.cls_names = cfg.cls_names
        # self.input_img_size = input_img_size
        self.split = split
        self.mask_type = mask_type

        # self.img_dir = os.path.join(self.data_root, self.mode)
        self.img_dir = os.path.join(self.data_root, self.split)

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
            height, width = self.cfg.img_width, self.cfg.img_height
        else:
            raise TypeError("Invalid type for variable index")

        img_info = self.imgs_info[img_id]

        input_img_size = (width, height)
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

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

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


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


# ------------------------ #
# Augmentation Fuction #
# ------------------------ #
class ToTensor(object):
    def __call__(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype("float32")


def get_train_augmentation():
    _transform = Compose([ToTensor()])
    return _transform


if __name__ == "__main__":
    from lib.datasets.make_datasets import make_data_loader

    cfg = CfgNode()
    cfg.task = "semantic_segm"
    cfg.cls_names = ["laptop", "tv"]
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CfgNode()
    cfg.train.dataset = "COCO2017Val"
    cfg.train.batch_size = 4
    cfg.train.num_workers = 2
    cfg.train.batch_sampler = ""

    dloader = make_data_loader(cfg, is_train=True)
    for iter, batch in enumerate(dloader):
        img, mask = batch["img"], batch["mask"]
