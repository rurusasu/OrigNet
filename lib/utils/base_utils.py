import os
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


file_ext = {
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


def GetImgFpsAndLabels(data_root: str):
    """指定したディレクトリ内の画像ファイルパス(Image file paths)とクラスラベルの一覧を取得する関数．

    Arg:
        data_root (str): 画像データが格納された親フォルダ
    Return:
        classes (list): クラス名のリスト
        class_to_idx (dict): クラス名と label_num を対応させる辞書を作成
        imgs (list): データパスと label_num を格納したタプルを作成．
                         例: [img_fp1, img_fp2, ...]
        targets (list): cls_num を格納したリスト
    """
    # train の子ディレクトリ名を教師ラベルとして設定
    classes = []
    class_to_idx = {}
    imgs = []
    msks = []
    targets = []
    for i, p in enumerate(glob(os.path.join(data_root, "*"))):
        cls_name = os.path.basename(p.rstrip(os.sep))
        # クラス名のリストを作成
        classes.append(cls_name)
        # クラス名と label_num を対応させる辞書を作成
        class_to_idx[cls_name] = i

        # クラス名ディレクトリ内の file_ext にヒットするパスを全探索
        if os.path.exists(os.path.join(p, "imgs")) and os.path.exists(
            os.path.join(p, "masks")
        ):
            # RGB 画像を探索
            for img_pth in glob(os.path.join(p, "imgs", "**"), recursive=True):
                if os.path.isfile(img_pth) and os.path.splitext(img_pth)[1] in file_ext:
                    # 画像データパスをリストに格納
                    imgs.append(img_pth)
                    # label_num をリストに格納
                    targets.append(i)

            # mask 画像を探索
            for msk_pth in glob(os.path.join(p, "masks", "**"), recursive=True):
                if os.path.isfile(msk_pth) and os.path.splitext(msk_pth)[1] in file_ext:
                    # マスク画像データパスをリストに格納
                    msks.append(msk_pth)
        else:
            # ディレクトリ内のすべての画像を読みだす．
            for img_pth in glob(os.path.join(p, "**"), recursive=True):
                if os.path.isfile(img_pth) and os.path.splitext(img_pth)[1] in file_ext:
                    # データパスと label_num を格納したタプルを作成
                    imgs.append(img_pth)
                    # label_num のみ格納したリストを作成
                    targets.append(i)

    return classes, class_to_idx, imgs, targets, msks


def LoadImgs(
    img_fps: List[str], img_id: int, msk_fps: List[str] = []
) -> Dict[np.ndarray, np.ndarray]:
    """
    画像パスのリストから、id で指定された画像を読みだす関数

    Args:
        img_fps (List[str]): 画像パスのリスト
        img_id (int): 読みだすパス番号
        msk_fps (List[str]): マスク画像パスのリスト

    Returns:
        imgs (Dict[img:np.ndarray, msk:np.ndarray]): 画像とそのマスク画像が保存された辞書
    """
    imgs = {}
    # 画像を読みだす
    img_fp = img_fps[img_id]
    img = Image.open(img_fp)
    # PIL -> OpenCV 型(numpy)に変換
    img = np.array(img, dtype=np.uint8)

    if msk_fps:
        msk_fp = msk_fps[img_id]
        msk = cv2.imread(msk_fp, 0)
    else:
        msk = []

    imgs["img"], imgs["msk"] = img, msk
    return imgs
