import os
from distutils.dir_util import copy_tree
import sys
from glob import glob
from typing import Dict, List, Literal

from torch.utils import data

sys.path.append("../../")

import cv2
import numpy as np
import pandas as pd
import skimage.io as io
from tqdm.contrib import tenumerate

from lib.visualizers.segmentation import visualize_np
from lib.utils.base_utils import LoadNdjson, WriteDataToNdjson

# --------------------------------------------- #
# Amazon Robotic Challenge 2017 Dataset #
# --------------------------------------------- #
# 中部大学 機能知覚 & ロボティクスグループ
# REF: http://mprg.jp/research/arc_dataset_2017_j


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


PixelValueToLabel = {
    "item1": [np.array([1], dtype=np.uint8), np.array([85, 0, 0], dtype=np.uint8)],
    "item2": [np.array([2], dtype=np.uint8), np.array([170, 0, 0], dtype=np.uint8)],
    "item3": [np.array([3], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8)],
    "item4": [np.array([4], dtype=np.uint8), np.array([0, 85, 0], dtype=np.uint8)],
    "item5": [np.array([5], dtype=np.uint8), np.array([85, 85, 0], dtype=np.uint8)],
    "item6": [np.array([6], dtype=np.uint8), np.array([170, 85, 0], dtype=np.uint8)],
    "item7": [np.array([7], dtype=np.uint8), np.array([255, 85, 0], dtype=np.uint8)],
    "item8": [np.array([8], dtype=np.uint8), np.array([0, 170, 0], dtype=np.uint8)],
    "item9": [np.array([9], dtype=np.uint8), np.array([85, 170, 0], dtype=np.uint8)],
    "item10": [np.array([10], dtype=np.uint8), np.array([170, 170, 0], dtype=np.uint8)],
    "item11": [np.array([11], dtype=np.uint8), np.array([255, 170, 0], dtype=np.uint8)],
    "item12": [np.array([12], dtype=np.uint8), np.array([0, 255, 0], dtype=np.uint8)],
    "item13": [np.array([13], dtype=np.uint8), np.array([85, 255, 0], dtype=np.uint8)],
    "item14": [np.array([14], dtype=np.uint8), np.array([170, 255, 0], dtype=np.uint8)],
    "item15": [np.array([15], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8)],
    "item16": [np.array([16], dtype=np.uint8), np.array([0, 0, 85], dtype=np.uint8)],
    "item17": [np.array([17], dtype=np.uint8), np.array([85, 0, 85], dtype=np.uint8)],
    "item18": [np.array([18], dtype=np.uint8), np.array([170, 0, 85], dtype=np.uint8)],
    "item19": [np.array([19], dtype=np.uint8), np.array([255, 0, 85], dtype=np.uint8)],
    "item20": [np.array([20], dtype=np.uint8), np.array([0, 85, 85], dtype=np.uint8)],
    "item21": [np.array([21], dtype=np.uint8), np.array([85, 85, 85], dtype=np.uint8)],
    "item22": [np.array([22], dtype=np.uint8), np.array([170, 85, 85], dtype=np.uint8)],
    "item23": [np.array([23], dtype=np.uint8), np.array([255, 85, 85], dtype=np.uint8)],
    "item24": [np.array([24], dtype=np.uint8), np.array([0, 170, 85], dtype=np.uint8)],
    "item25": [np.array([25], dtype=np.uint8), np.array([85, 170, 85], dtype=np.uint8)],
    "item26": [
        np.array([26], dtype=np.uint8),
        np.array([170, 170, 85], dtype=np.uint8),
    ],
    "item27": [
        np.array([27], dtype=np.uint8),
        np.array([255, 170, 85], dtype=np.uint8),
    ],
    "item28": [np.array([28], dtype=np.uint8), np.array([0, 255, 85], dtype=np.uint8)],
    "item29": [np.array([29], dtype=np.uint8), np.array([85, 255, 85], dtype=np.uint8)],
    "item30": [
        np.array([30], dtype=np.uint8),
        np.array([170, 255, 85], dtype=np.uint8),
    ],
    "item31": [
        np.array([31], dtype=np.uint8),
        np.array([255, 255, 85], dtype=np.uint8),
    ],
    "item32": [np.array([32], dtype=np.uint8), np.array([0, 0, 170], dtype=np.uint8)],
    "item33": [np.array([33], dtype=np.uint8), np.array([85, 0, 170], dtype=np.uint8)],
    "item34": [np.array([34], dtype=np.uint8), np.array([170, 0, 170], dtype=np.uint8)],
    "item35": [np.array([35], dtype=np.uint8), np.array([255, 0, 170], dtype=np.uint8)],
    "item36": [np.array([36], dtype=np.uint8), np.array([0, 85, 170], dtype=np.uint8)],
    "item37": [np.array([37], dtype=np.uint8), np.array([85, 85, 170], dtype=np.uint8)],
    "item38": [
        np.array([38], dtype=np.uint8),
        np.array([170, 85, 170], dtype=np.uint8),
    ],
    "item39": [
        np.array([39], dtype=np.uint8),
        np.array([255, 85, 170], dtype=np.uint8),
    ],
    "item40": [
        np.array([40], dtype=np.uint8),
        np.array([0, 170, 170], dtype=np.uint8),
    ],
}


class ARCDatasetTransformer(object):
    def __init__(
        self, data_root: str, split: Literal["train", "val", "test"] = "train"
    ) -> None:
        super(ARCDatasetTransformer, self).__init__()
        self.ds_pth = os.path.join(data_root, split)
        self.img_dir_pth = os.path.join(self.ds_pth, "rgb")
        self.ann_dir_pth = os.path.join(self.ds_pth, "annotations")
        self.ann_json_pth = os.path.join(self.ann_dir_pth, "instances_train2017.json")
        self.seg_img_root_dir = os.path.join(self.ds_pth, "seg_instance")
        self.seg_img_conlbl_dir = os.path.join(self.ds_pth, "temp")

        # json ファイルが存在しない場合
        if not os.path.exists(self.ann_json_pth):
            if not os.path.exists(self.ann_dir_pth):
                print(f"{self.ann_dir_pth} にディレクトリが存在しません．ディレクトリを再度作成します．")
                os.makedirs(self.ann_dir_pth, exist_ok=True)

            print(f"{self.ann_json_pth} にファイルが存在しなかったので，json ファイルを再度作成します．")
            if not os.path.exists(self.seg_img_conlbl_dir):
                print("json ファイル作成のための temp ディレクトリが存在しません．ディレクトリを再度作成します．")
                self._CreateContinuousLabelImage(
                    save_img=True, save_root_dir=self.seg_img_conlbl_dir
                )

            print("jsonファイルの作成を開始します．")
            self.CreateSource(self.ann_json_pth)

    def CreateSource(self, wt_json_pth: str):
        """アノテーションデータを JSON ファイルに出力する関数．

        Args:
            wt_json_pth (str): データを書き込む `JSON` ファイルへのパス．
        """
        image_id = []
        # img_dir の画像ファイルパスを検索．
        # RGB 画像を探索
        for id, img_pth in tenumerate(
            glob(os.path.join(self.img_dir_pth, "**"), recursive=True)
        ):
            # パスがファイルかつ拡張子が画像のものか判定．
            if os.path.isfile(img_pth) and os.path.splitext(img_pth)[1] in file_ext:
                image_id.append(os.path.splitext(os.path.basename(img_pth))[0])

        id = 0
        for _, img_id in tenumerate(image_id):
            # 画像のファイル名と同じディレクトリを seg_instance から検索．
            seg_img_dir = os.path.join(self.seg_img_conlbl_dir, img_id + "_s")
            for seg_pth in glob(os.path.join(seg_img_dir, "**"), recursive=True):

                # パスがファイルかつ拡張子が画像のものか判定．
                if os.path.isfile(seg_pth) and os.path.splitext(seg_pth)[1] in file_ext:
                    seg_img = io.imread(seg_pth)
                    # pixcel 座標: ndarray[0, 1, ...][0, 1, ...] -> (0, 0), (1, 1), ..
                    pix_x, pix_y = np.where(seg_img > 0)  # 画素値 > 0 の位置を取得．
                    # 画素値の値を label に変換
                    pv = seg_img[pix_x[0]][pix_y[0]]  # 1つの画素値を取得
                    category_id = [
                        v[0]
                        for _, v in PixelValueToLabel.items()
                        if np.allclose(v[0], pv)
                    ][0]

                    # すべてのデータが揃っているか判定．
                    if (image_id is not None) and (category_id is not None):
                        id += 1
                        # 画像情報 & アノテーション情報
                        source = {
                            "img_info": {
                                "image_id": img_id,
                                "id": id,
                                "imgs": {"file_name": str(img_id) + ".png", "id": id},
                            },
                            "anns": {
                                "category_id": category_id.max().tolist(),
                                "segmentation": [pix_x.tolist(), pix_y.tolist()],
                            },
                        }

                        WriteDataToNdjson(source, wt_json_pth)
                        # 不要な変数を削除
                        del seg_img, pix_x, pix_y, pv

    def _CreateContinuousLabelImage(
        self,
        img_size: Dict = {"w": 640, "h": 480},
        save_img: bool = False,
        save_root_dir: str = "./temp",
    ):
        """
        RGB の3チャネルで保存されている segmentation label (例: [170, 0, 0]) を
        グレー階調の連続値の label (例: [2]) に変換する関数．

        Args:
            save_img (bool, optional): 変換した label image を保存するか. Defaults to False.
            save_root_dir (str, optional): 変換した label image の保存先のルートディレクトリ. Defaults to ".".
        """
        seg_img_root_dir = self.seg_img_root_dir
        # もし，ContinuousLabel に変更した画像を保存する場合．
        if save_img:
            # ---- root dir ---- #
            # もし，すでにディレクトリが存在する場合
            if os.path.exists(save_root_dir):
                print("すでにディレクトリが存在するため，一度ディレクトリを削除します．")
                # ディレクトリ内のすべてのデータを削除
                os.system("rm -rf {}".format(save_root_dir))
                # ディレクトリ作成
                os.makedirs(save_root_dir)
            # もし，ディレクトリが存在しない場合
            else:
                # ディレクトリ作成
                os.makedirs(save_root_dir)
            print("画像保存用に一度すべての画像をコピーします．")
            # save_root_dir 側にすべての画像ファイルをコピー
            # REF: https://qiita.com/supersaiakujin/items/12451cd2b8315fe7d054
            copy_tree(seg_img_root_dir, save_root_dir)
            print("コピーが完了しました．")
        # 保存しない場合
        else:
            save_root_dir = seg_img_root_dir

        for seg_pth in glob(os.path.join(save_root_dir, "**"), recursive=True):
            # パスがファイルかつ拡張子が画像のものの場合．
            if os.path.isfile(seg_pth) and os.path.splitext(seg_pth)[1] in file_ext:
                # ラベル画像の読み出し．
                seg_img = io.imread(seg_pth)

                print(f"{os.path.basename(seg_pth)} を gray scale label に変換します．")
                gray_img = self.__RGBLabelToContinuousLabel(seg_img)

                # Resize: [H, W, C] -> [H', W', C]
                # 変数が，[W, H] で与える点に注意
                gray_img = cv2.resize(gray_img, (img_size["w"], img_size["h"]))

                # 作成した画像を保存する場合
                if save_img:
                    print("画像を上書き保存します．")
                    cv2.imwrite(seg_pth, gray_img)
                else:
                    visualize_np(org_segmentation=seg_img, ContinuousLabel=gray_img)

    def __RGBLabelToContinuousLabel(self, RGB: np.ndarray) -> np.ndarray:
        """
        RGB画像の画素値で表されたラベルを，グレースケール画像の連続値のラベルに変換する関数．

        Arg:
            RGB(np.ndarray): ([H, W, C]) の画像．

        Return:
            conlbl(np.ndarray): 連続値のラベルに変換した ([H, W]) の画像．
        """
        conlbl = None
        conlbl = RGB.copy()

        # pixel value > 0 の 位置の画素値を取得
        #  セグメンテーション領域内にオブジェクト同士が重なっている場所がある場合
        # seg_img = [R, G, B, A] で保存されているため，後の r, g, b = cv2.split() がエラーになる．
        if conlbl.shape[2] > 3:
            # conlbl = np.delete(conlbl, axis=4)
            conlbl = conlbl[:, :, :3]

            # 白である
            w_pix = (
                (conlbl[..., 0] == 255)
                & (conlbl[..., 1] == 255)
                & (conlbl[..., 2] == 255)
            )
            # 黒である
            bk_pix = (
                (conlbl[..., 0] == 0) & (conlbl[..., 1] == 0) & (conlbl[..., 2] == 0)
            )

            # 白でない
            not_w_pix = np.logical_not(w_pix)
            not_bk_pix = np.logical_not(bk_pix)
            pix = np.logical_and(not_w_pix, not_bk_pix)
            # orig_shape = (pix.shape[0], pix.shape[1], -1)

            pv = conlbl[pix][0]

            # lbl_img = lbl_img.reshape(orig_shape)
            # point = np.nonzero(lbl_img)
            # pv = lbl_img[point[0][0]][point[1][0]]
            conlbl[w_pix] = pv
            from matplotlib import pyplot as plt

            fig = plt.figure()
            plt.imshow(conlbl)
            plt.show()

        r, g, b = cv2.split(conlbl)
        pv = np.array([r.max(), g.max(), b.max()])
        lbl_value = [
            v[0] for _, v in PixelValueToLabel.items() if np.allclose(v[1], pv)
        ][0]

        # RGB Seg_img をグレー化
        conlbl = cv2.cvtColor(conlbl, cv2.COLOR_RGB2GRAY)
        _, conlbl = cv2.threshold(conlbl, 1, 1, cv2.THRESH_BINARY)
        conlbl = conlbl * lbl_value

        return conlbl


class ARCDataset(object):
    def __init__(self, f_json_pth: str) -> None:
        super(ARCDataset, self).__init__()
        self.f_json_pth = f_json_pth
        # path に .json が含まれていなければ追加
        if ".json" not in os.path.splitext(self.f_json_pth)[-1]:
            self.f_json_pth = self.f_json_pth + ".json"

        if not os.path.exists(self.f_json_pth) or not os.path.isfile(self.f_json_pth):
            raise ValueError("`{}` is invalid.".format(self.f_json_pth))
        else:
            print("json ファイルを読みだします．")
            # source = LoadNdjson(self.f_json_pth)
            # result = []

            self.df = pd.read_json(self.f_json_pth, orient="record", lines=True)
            self.df = pd.json_normalize(self.df.to_dict("records"), sep="_")

    def getCatIds(self, catNms: str) -> np.ndarray:
        """
        カテゴリの名前 (str) を対応するラベルの値 ndarray[int] に変換する関数．
        例: "item10" -> ndarray([10], dtype=uint8)

        Args:
            catNms (str): ARCDatasetのカテゴリ名．

        Returns:
            (ndarray): ラベル値．
        """
        return PixelValueToLabel[catNms][0]

    def getImgIds(self, catIds: np.ndarray) -> List[int]:
        """ラベル値 ndarray[int] と一致するデータ番号 (id) のリストを返す関数．

        Args:
            catIds (List[int]): ラベル値の ndarray 配列．

        Returns:
            List[int]: データ番号 (id) のリスト．例: [1, 2, 3, ]
        """
        df = self.df.query(f"anns_category_id in {catIds}")
        return df["img_info_id"].tolist()

    def loadImgs(self, imgIds: List[int]):
        """

        Args:
            imgIds (List[int]): [description]
        """

    def get_df(self):
        return self.df


if __name__ == "__main__":
    import pprint
    import sys

    sys.path.append("../../")
    from lib.config.config import pth

    root = os.path.join(pth.DATA_DIR, "ARCdataset_png")
    json_fp = os.path.join(
        pth.DATA_DIR,
        "ARCdataset_png",
        "train",
        "annotations",
        "instances_train2017.json",
    )

    ds = ARCDatasetTransformer(root, split="train")
    # ds.CreateSource(json_fp)
    # save_root_dir = os.path.join(pth.DATA_DIR, "test")
    # ds.CreateContinuousLabelImage(save_img=False, save_root_dir=save_root_dir)

    # seg_pth = "/mnt/d/My_programing/OrigNet/data/ARCdataset_png/train/con_lbl/2017-002-1_s/2017-002-1_s_3.png"
    # seg_pth = "/mnt/d/My_programing/OrigNet/data/ARCdataset_png/train/con_lbl/2017-002-1_s/2017-002-1_s_32.png"
    # seg_img = io.imread(seg_pth)
    # ds.RGBLabelToContinuousLabel(seg_img)

    dataset = ARCDataset(json_fp)
    df = dataset.get_df()
    pprint.pprint(df)

    catNms = "item1"
    catIds = dataset.getCatIds(catNms)
    print(catIds)

    imgIds = dataset.getImgIds(catIds)
    print(imgIds)
