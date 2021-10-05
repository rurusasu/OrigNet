import os
import random
from typing import List, Literal, Union

import cv2
import numpy as np
import skimage.io as io
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

### For visualizing the outputs ###
import matplotlib.pyplot as plt


def getClassName(classID: int, cats: dict):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


def FilterDataset(
    root_dir,
    classes: Union[List[str], None] = None,
    mode: Literal["train", "val", "test"] = "train",
):
    """フィルタしたクラスのオブジェクトが映る画像をすべて読みだす関数

    Args:
        root_dir (str): データセットの root ディレクトリ．
        classes (Union(List[str], None), optional): 抽出するクラス名のリスト. Defaults to None.
        mode (str, optional): 読みだすデータセットの種類（'train' or 'val', or 'test'）. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    # initialize COCO api for instance annotations
    annFile = "{}/annotations/instances_{}2017.json".format(root_dir, mode)
    coco = COCO(annFile)

    images = []
    if classes != None:
        # リスト内の個々のクラスに対してイテレートする
        for className in classes:
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


def getImage(imgObj, img_folder: str, input_img_size: tuple):
    # Read and normalize an image
    train_img = io.imread(os.path.join(img_folder, imgObj["file_name"])) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_img_size)
    if (
        len(train_img.shape) == 3 and train_img.shape[2] == 3
    ):  # If it is a RGB 3 channel image
        return train_img
    else:  # 白黒の画像を扱う場合は、次元を3にする
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imgObj, classes, coco, catIds, input_img_size):
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_img_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_img_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_img_size[0], input_img_size[1], 1)
    return train_mask


def getBinaryMask(imgObj, coco, catIds, input_img_size):
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


class COCODataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        classes: Union[List[str], None] = None,
        input_img_size: tuple = (224, 224),
        mode: Literal["train", "val", "test"] = "train",
        mask_type: Literal["binary", "normal"] = "binary",
        augmentation: Union[Compose, None] = None,
    ):
        self.root_dir = root_dir
        self.classes = classes
        self.input_img_size = input_img_size
        self.mode = mode
        self.mask_type = mask_type

        self.img_dir = os.path.join(self.root_dir, self.mode)

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
            self.root_dir, self.classes, self.mode
        )
        self.catIds = self.coco.getCatIds(catNms=self.classes)

        # Data Augmentation
        self.augmentation = augmentation

    def __getitem__(self, idx):
        img_info = self.imgs_info[idx]

        ### Retrieve Image ###
        img = getImage(
            imgObj=img_info, img_folder=self.img_dir, input_img_size=self.input_img_size
        )

        ### Create Mask ###
        if self.mask_type == "binary":
            mask = getBinaryMask(img_info, self.coco, self.catIds, self.input_img_size)

        elif self.mask_type == "normal":
            mask = getNormalMask(
                img_info, self.classes, self.coco, self.catIds, self.input_img_size
            )

        if self.augmentation:
            img = self.augmentation(img)
            mask = self.augmentation(mask)

        return img, mask

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
def get_train_augmentation():
    _transform = Compose([ToTensor()])
    return _transform


if __name__ == "__main__":
    dataDir = "../../../data/COCOdataset2017/"
    class_names = ["laptop", "tv"]
    mode = "train"
    input_img_size = (224, 224)
    mask_type = "normal"

    train_dataset = COCODataset(
        root_dir=dataDir,
        classes=class_names,
        input_img_size=input_img_size,
        mask_type=mask_type,
        augmentation=get_train_augmentation(),
    )
    valid_dataset = COCODataset(
        root_dir=dataDir,
        classes=class_names,
        input_img_size=input_img_size,
        mode="val",
        mask_type="normal",
        augmentation=get_train_augmentation(),
    )

    # training
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    ENCODER = "efficientnet-b0"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = "softmax2d"
    DEVICE = "cuda"

    n_classes = (
        1 if len(class_names) == 1 else (len(class_names) + 1)
    )  # case for binary and multiclass segmentation

    # create segmentation model with pretrained encoder
    print("model loading ...")
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=n_classes,  # len(CLASSES),
        activation=ACTIVATION,
    )
    print("model created !")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    BATCH_SIZE = 8
    SHUFFLE = True
    NUM_WORKERS = 2

    print("creating Data Loader ...")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    print("Data Loader cteated !")

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.001)])

    print("creating epoch function ...")
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    print("Epoch function created !")

    """
    for i in range(len(train_dataset)):
        img, mask = train_dataset[i]
        # plt.imshow(mask)
        # plt.show()
        visualize(image=img, mask=mask)
    """

    max_score = 0

    # train accurascy, train loss, val_accuracy, val_loss をグラフ化できるように設定．
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []

    print("Run training！")
    for i in range(0, 20):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

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
