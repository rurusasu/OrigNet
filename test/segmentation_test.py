import os
import sys
import random
from typing import List, Literal, Type, Union

sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import skimage.io as io
import torch
import torch.utils.data as data
from torch._C import TensorType
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose
from pycocotools.coco import COCO
from yacs.config import CfgNode

from lib.config.config import pth
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.datasets.samplers import ImageSizeBatchSampler, IterationBasedBatchSampler
from lib.datasets.transforms import make_transforms
from lib.models.make_network import make_network


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
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
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


def getImage(imgObj, img_folder: str, input_img_size: tuple) -> np.ndarray:
    # Read and normalize an image
    img = io.imread(os.path.join(img_folder, imgObj["file_name"]))
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
    # del annIds
    # gc.collect()

    cats = coco.loadCats(catIds)
    mask = np.zeros(input_img_size, dtype=np.uint8)
    class_names = []
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = cls_names.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_img_size)
        mask = np.maximum(new_mask, mask, dtype=np.uint8)
        class_names.append(className)

    # del anns, cats
    # gc.collect()
    # Add extra dimension for parity with train_img size [X * X * 3]
    mask = mask.reshape(input_img_size[0], input_img_size[1], 1)

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
        mask = np.maximum(new_mask, mask, dtype=np.uint8)

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
        transforms: Union[Compose, None] = None,
    ):
        self.cfg = cfg
        self.data_root = os.path.join(pth.DATA_DIR, data_root)
        self.cls_names = cfg.cls_names
        self.split = split
        self.mask_type = mask_type

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
        del img_id

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

        # del img_info, input_img_size
        # gc.collect()

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)

        # ndarray -> tensor
        # img = torch.from_numpy(img.astype(np.float32)).clone()
        # mask = torch.from_numpy(mask.astype(np.float32)).clone()

        return img, mask

    def __len__(self):
        return self.dataset_size


_dataset_factory = {"semantic_segm": SegmentationDataset}


def make_dataset(
    cfg: CfgNode,
    dataset_name: str,
    transforms: Type[Union[None, TensorType]] = None,
    is_train: bool = True,
):
    """
    `DatasetCatalog` から `dataset_name` を参照し，そこに保存されている情報をもとにデータセットを作成する関数

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        dataset_name (str): ロードするデータセット名
        transforms (torchvision.transforms): データ拡張に使用するtorchvisionのクラス．default to None.
        is_train (bool): 訓練用データセットか否か．default to True.
    """
    args = DatasetCatalog.get(dataset_name)
    dataset = _dataset_factory[cfg.task]
    args["cfg"] = cfg
    del args["id"]

    # args["data_root"] = os.path.join(pth.DATA_DIR, args["data_root"])
    if transforms is not None:
        args["transforms"] = transforms
    # args["split"] = "train" if is_train else "test"
    dataset = dataset(**args)

    return dataset


def _make_data_sampler(dataset, shuffle: bool):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _make_batch_data_sampler(
    sampler: Sampler,
    batch_size: int,
    drop_last: bool,
    max_iter: int,
    strategy: Literal[
        "image_size",
    ] = "image_size",
):
    """
    イタレーションごとにデータセットからデータをサンプリングする際に行う処理を決定する関数

    Args:
        sampler (Sampler): データセットからデータをサンプリングする際の処理を自動化するクラス．
        batch_size (int): バッチサイズ．
        drop_last (bool): サンプリングしきれなかった余りを切り捨てるか．
        max_iter (int): イテレーションの最大値．
        strategy (Literal[str, optional): 特殊な `batch_sampler` を使用する場合に設定する.
                                                    Defaults to "image_size".

    Returns:
        [type]: [description]
    """
    if strategy == "image_size":
        batch_sampler = ImageSizeBatchSampler(
            sampler, batch_size, drop_last, 256, 480, 640
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last
        )
        if max_iter != -1:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iter)

    return batch_sampler


def _worker_init_fn(worker_id):
    """
    workerの初期化時に乱数のシードを個別に設定する関数
    これにより、workerの分だけforkするときに同一のnumpyのRandom Stateの状態がコピーされるため生じる入力データの重複を避けることができる。
    REF: https://qiita.com/kosuke1701/items/14cd376e024f86e57ff6
    """
    # np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def make_data_loader(
    cfg: CfgNode,
    is_train: bool = True,
    is_distributed: bool = False,
    max_iter: int = -1,
):
    """
    データローダーを作成する関数．

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        is_train (bool): 訓練用データセットか否か．Defaults to True.
        is_distributed (bool): データをシャッフルしたものをテストに使用するか．Defaults to False.
        max_iter (int): イテレーションの最大値．Defaults to -1.
    """
    if is_train:
        if (
            "train" not in cfg
            and "dataset" not in cfg.train
            and "batch_size" not in cfg.train
            and "num_workers" not in cfg.train
            and "batch_sampler" not in cfg.train
        ):
            raise ("The required parameter for 'make_data_loader' has not been set.")
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
        num_workers = cfg.train.num_workers
        batch_sampler = cfg.train.batch_sampler
    else:
        if (
            "test" not in cfg
            and "dataset" not in cfg.test
            and "batch_size" not in cfg.test
            and "num_workers" not in cfg.test
            and "batch_sampler" not in cfg.test
        ):
            raise ("The required parameter for `make_data_loader` has not been set.")
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False
        num_workers = cfg.test.num_workers
        batch_sampler = cfg.test.batch_sampler

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(
        cfg, dataset_name=dataset_name, transforms=transforms, is_train=is_train
    )
    sampler = _make_data_sampler(dataset, shuffle)
    batch_sampler = _make_batch_data_sampler(
        sampler, batch_size, drop_last, max_iter, batch_sampler
    )

    # 引数: pin_memory について
    # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
    )

    return data_loader


cfg = CfgNode()
cfg.cls_names = ["laptop", "tv"]
cfg.task = "semantic_segm"
cfg.network = "smp"
cfg.model = "unetpp"
cfg.encoder_name = "efficientnet-b0"
cfg.img_width = 224
cfg.img_height = 224
cfg.ep_iter = -1
cfg.train = CfgNode()
cfg.train.dataset = "COCO2017Val"
cfg.train.batch_size = 4
cfg.train.batch_sampler = ""
cfg.train.num_workers = 2
cfg.test = CfgNode()
cfg.test.dataset = "COCO2017Val"
cfg.test.batch_size = 20
cfg.test.num_workers = 2
cfg.test.batch_sampler = ""


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
val_loader = make_data_loader(cfg, is_train=False)

cfg.num_classes = len(train_loader.dataset.cls_names)
model = make_network(cfg)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001)])

train_epoch = smp.utils.train.TrainEpoch(
    model, loss=loss, metrics=metrics, optimizer=optimizer, device=DEVICE, verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
)


# train model for 100 epochs
max_score = 0

# train accurascy, train loss, val_accuracy, val_loss をグラフ化できるように設定．
x_epoch_data = []
train_dice_loss = []
train_iou_score = []
valid_dice_loss = []
valid_iou_score = []

for i in range(0, 20):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)

    x_epoch_data.append(i)
    train_dice_loss.append(train_logs["dice_loss"])
    train_iou_score.append(train_logs["iou_score"])
    valid_dice_loss.append(valid_logs["dice_loss"])
    valid_iou_score.append(valid_logs["iou_score"])

    # do something (save model, change lr, etc.)
    if max_score < valid_logs["iou_score"]:
        max_score = valid_logs["iou_score"]
        torch.save(model, "./best_model_quita.pth")
        print("Model saved!")

    if i == 25:
        optimizer.param_groups[0]["lr"] = 1e-5
        print("Decrease decoder learning rate to 1e-5!")

    if i == 50:
        optimizer.param_groups[0]["lr"] = 5e-6
        print("Decrease decoder learning rate to 5e-6!")

    if i == 75:
        optimizer.param_groups[0]["lr"] = 1e-6
        print("Decrease decoder learning rate to 1e-6!")
