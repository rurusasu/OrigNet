import os
import sys
import random
from typing import Dict, List, Literal, Union

sys.path.append("../../../")

import cv2
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from yacs.config import CfgNode


def getClassName(classID: int, cats: dict):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


def FilterCOCODataset(
    data_root,
    cls_names: Union[List[str], None] = None,
    split: Literal["train", "val", "test"] = "train",
):
    """フィルタリングしたクラスのオブジェクトが映る画像をすべて読みだす関数

    Args:
        data_root (str): データセットの root ディレクトリ．
        cls_names (Union(List[str], None), optional): 抽出するクラス名のリスト. Defaults to None.
        split (Literal["train", "val", "test"], optional): 読みだすデータセットの種類（'train' or 'val' or 'test'). Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    # initialize COCO api for instance annotations
    # annFile = "{}/annotations/instances_{}2017.json".format(data_root, split)
    annFile = "{}/{}/annotations/instances_{}2017.json".format(data_root, split, split)
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
        cls_names = []
        catIDs = coco.getCatIds()
        cats = coco.loadCats(catIDs)
        for catID in catIDs:
            cls_names.append(getClassName(classID=catID, cats=cats))
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

    return unique_images, dataset_size, coco, cls_names


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


def getCOCONormalMask(imgObj, cls_names, coco, catIds, input_img_size):
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

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

    # Add extra dimension for parity with train_img size [X * X * 3]
    # mask = mask.reshape(input_img_size[0], input_img_size[1], 1)

    # すべての
    return mask, class_names


def getCOCOBinaryMask(imgObj, coco, catIds, input_img_size) -> np.ndarray:
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from lib.datasets.make_datasets import make_data_loader
    from lib.visualizers.segmentation import visualize

    conf = CfgNode()
    conf.task = "semantic_segm"
    conf.cls_names = ["laptop", "tv"]
    conf.img_width = 400
    conf.img_height = 400
    conf.train = CfgNode()
    conf.train.dataset = "COCO2017Val"
    conf.train.batch_size = 4
    conf.train.num_workers = 1
    conf.train.batch_sampler = ""
    conf.test = CfgNode()
    conf.test.dataset = "COCO2017Val"
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
