from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


def load_img(
    img_fps: List[str], img_id: int, msk_fps: List[str] = None
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

    if msk_fps is not None:
        msk_fp = msk_fps[img_id]
        msk = cv2.imread(msk_fp, 0)
    else:
        msk = []

    imgs["img"], imgs["msk"] = img, msk
    return imgs
