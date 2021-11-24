import os
import random
import sys
from distutils.dir_util import copy_tree
from glob import glob
from typing import Dict, List, Literal, Union

sys.path.append("../../")
sys.path.append("../../../")

import cv2
import numpy as np
import pandas as pd
import scipy.stats as stats
import skimage.io as io
from tqdm.contrib import tenumerate

from lib.visualizers.segmentation import visualize_np
from lib.utils.base_utils import WriteDataToNdjson

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

ClassLabel = [
    {"name": "item1", "id": 1},
    {"name": "item2", "id": 2},
    {"name": "item3", "id": 3},
    {"name": "item4", "id": 4},
    {"name": "item5", "id": 5},
    {"name": "item6", "id": 6},
    {"name": "item7", "id": 7},
    {"name": "item8", "id": 8},
    {"name": "item9", "id": 9},
    {"name": "item10", "id": 10},
    {"name": "item11", "id": 11},
    {"name": "item12", "id": 12},
    {"name": "item13", "id": 13},
    {"name": "item14", "id": 14},
    {"name": "item15", "id": 15},
    {"name": "item16", "id": 16},
    {"name": "item17", "id": 17},
    {"name": "item18", "id": 18},
    {"name": "item19", "id": 19},
    {"name": "item20", "id": 20},
    {"name": "item21", "id": 21},
    {"name": "item22", "id": 22},
    {"name": "item23", "id": 23},
    {"name": "item24", "id": 24},
    {"name": "item25", "id": 25},
    {"name": "item26", "id": 26},
    {"name": "item27", "id": 27},
    {"name": "item28", "id": 28},
    {"name": "item29", "id": 29},
    {"name": "item30", "id": 30},
    {"name": "item31", "id": 31},
    {"name": "item32", "id": 32},
    {"name": "item33", "id": 33},
    {"name": "item34", "id": 34},
    {"name": "item35", "id": 35},
    {"name": "item36", "id": 36},
    {"name": "item37", "id": 37},
    {"name": "item38", "id": 38},
    {"name": "item39", "id": 39},
    {"name": "item40", "id": 40},
]


class ARCDatasetTransformer(object):
    def __init__(
        self, data_root: str, split: Literal["train", "val", "test"] = "train"
    ) -> None:
        super(ARCDatasetTransformer, self).__init__()
        self.ds_pth = os.path.join(data_root, split)
        self.img_dir_pth = os.path.join(self.ds_pth, "rgb")
        self.ann_dir_pth = os.path.join(self.ds_pth, "annotations")
        self.ann_json_pth = os.path.join(
            self.ann_dir_pth, "instances_{}2017.json".format(split)
        )
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
                    # pix_x, pix_y = np.where(seg_img > 0)  # 画素値 > 0 の位置を取得．
                    # 画素値の値を label に変換
                    # pv = seg_img[pix_x[0]][pix_y[0]]  # 1つの画素値を取得
                    # pv = seg_img.max()
                    # category_id = [
                    #     v[0]
                    #     for _, v in PixelValueToLabel.items()
                    #     if np.allclose(v[0], pv)
                    # ][0]

                    """
                    [Pythonのパスの最後の部分だけを手に入れる方法](https://www.fixes.pub/program/715603.html)
                    """
                    catIds = os.path.basename(os.path.normpath(seg_pth)).split("_")[-1]
                    # 15.png -> 15
                    catIds = os.path.splitext(catIds)[0]

                    # すべてのデータが揃っているか判定．
                    if (image_id is not None) and (catIds is not None):
                        id += 1
                        # 画像情報 & アノテーション情報
                        source = {
                            "image_id": img_id,
                            "id": id,
                            "file_name": str(img_id) + ".png",
                            "category_id": int(catIds),
                            "anno_file_name": seg_pth.replace(self.ds_pth + os.sep, ""),
                        }

                        WriteDataToNdjson(source, wt_json_pth)
                        # 不要な変数を削除
                        del seg_img

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
        # conlbl = None
        conlbl = RGB.copy()

        # pixel value > 0 の 位置の画素値を取得

        if conlbl.shape[2] > 3:
            # conlbl = np.delete(conlbl, axis=4)
            conlbl = conlbl[:, :, :3]

        # セグメンテーション領域内にオブジェクト同士が重なっている場所がある場合
        # その領域だけ，[255, 255, 255] の画素値を持つのでラベル変換の際にエラーが生じる．
        # そこで，[255, 255, 255] の画素値をその他のセグメンテーション領域の画素値と同じ値に置き換える．
        # REF: https://teratail.com/questions/100301

        # 白である
        w_pix = (
            (conlbl[..., 0] == 255) & (conlbl[..., 1] == 255) & (conlbl[..., 2] == 255)
        )
        # 黒である
        bk_pix = (conlbl[..., 0] == 0) & (conlbl[..., 1] == 0) & (conlbl[..., 2] == 0)
        not_w_pix = np.logical_not(w_pix)  # 白でない
        not_bk_pix = np.logical_not(bk_pix)  # 黒でない
        # 白でない かつ 黒でない 領域からラベルの画素値を抽出
        pix = np.logical_and(not_w_pix, not_bk_pix)

        # 抽出した画素値の中の最頻値をラベルの画素値と仮定．
        # オクルージョン領域(255, 255, 255) と 背景(0, 0, 0) は外しているので，
        # この方法で，ノイズがあったとしても正しい画素値を抽出できるはず．
        # REF: https://python.atelierkobato.com/mode/
        x = conlbl[pix]
        pv, _ = stats.mode(x, axis=0)

        # 画素値を置き換え
        conlbl[w_pix] = pv
        # conlbl[w_pix] = 1

        # -------------------------------------
        # デバッグの際の画像表示用
        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # plt.imshow(conlbl)
        # plt.show()
        # -------------------------------------

        # lbl_value = [v[0] for _, v in PixelValueToLabel.items() if np.allclose(v[1], pv)][0]

        # RGB Seg_img をグレー化
        conlbl = cv2.cvtColor(conlbl, cv2.COLOR_RGB2GRAY)
        _, conlbl = cv2.threshold(conlbl, 1, 1, cv2.THRESH_BINARY)
        # conlbl = conlbl * lbl_value

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

            self.df = pd.read_json(self.f_json_pth, orient="record", lines=True)
            self.df = pd.json_normalize(self.df.to_dict("records"), sep="_")

    def getCatIds(self, catNms: Union[None, str, List[str]] = None) -> List[int]:
        """
        カテゴリの名前 (str) を対応するラベルの値 ndarray[int] に変換する関数．
        例: "item10" -> ndarray([10], dtype=uint8)

        Args:
            catNms (Union[None, str, List[str]]):
            ARCDatasetのカテゴリ名．
            Noneの場合，すべてのカテゴリ名を読み出す．
            default to None.

        Raises:
            ValueError: catNms should be given as `str` or `list` type.

        Returns:
            List[int]: ラベル値．
        """
        if catNms is None:
            items = []
            for i in range(len(ClassLabel)):
                items.append(ClassLabel[i]["id"])
        elif isinstance(catNms, list):
            items = []
            for k in catNms:
                for i in range(len(ClassLabel)):
                    if ClassLabel[i]["name"] == k:
                        items.append(ClassLabel[i]["id"])
        elif isinstance(catNms, str):
            for i in range(len(ClassLabel)):
                if ClassLabel[i]["name"] == catNms:
                    items = [ClassLabel[i]["id"]]
        else:
            raise ValueError("catNms should be given as `str` or `list` type.")
        return items

    def getImgIds(self, catIds: Union[None, List[int]] = None) -> List[int]:
        """
        カテゴリid `List[int]` と一致するデータ番号 (id) のリストを返す関数．

        Args:
            catIds (Union[None, List[int]], optional): 読み出したい画像のカテゴリid. Defaults to None.

        Returns:
            List[int]: データ番号 (id) のリスト．例: [1, 2, 3, ]
        """
        if catIds is None:
            df = self.df
        elif isinstance(catIds, list):
            # pandas.DataFrameの行を条件で抽出するquery
            # REF: https://note.nkmk.me/python-pandas-query/
            df = self.df.query(f"category_id in {catIds}")
        else:
            raise ValueError("catIds should be given as `None` or `list` type.")
        return df["id"].tolist()

    def getAnnIds(self, imgIds: List[int]) -> List[Dict[str, str]]:
        """画像に対応するマスク画像

        Args:
            imgIds (List[int]): データセットないの画像に与えられた一意の id
        """
        # pandas.DataFrameの行を条件で抽出するquery
        # REF: https://note.nkmk.me/python-pandas-query/
        df = self.df.query(f"id in {imgIds}")

        df = df[["anno_file_name"]]
        # DataFrame -> Dict
        # REF: https://note.nkmk.me/python-pandas-to-dict/
        # df = df.to_dict(orient="records")
        df = df["anno_file_name"].index.values
        # ndarray -> list
        df = df.tolist()
        return df

    def loadAnns(
        self, annIds: Union[dict, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """アノテーション情報を読み出して，list 形式で返す関数．

        Args:
            annIds (Union[dict, List[Dict[str, str]]]): 読み出したいアノテーション情報の番号(df の index 番号に相当)．

        Raises:
            ValueError: ValueError: annIds should be given as `dict` or `list` type.

        Returns:
            List[Dict[str, str]]: アノテーション情報のリスト．
            {
                "anno_file_name": filepath,
                "category_id": int
            }
        """
        if isinstance(annIds, dict):
            items = self.df.query(f"anno_file_name in {annIds.values()}")
        elif isinstance(annIds, list):
            items = []
            for annId in annIds:
                df = self.df.loc[
                    self.df.index[annId], ["anno_file_name", "category_id"]
                ]
                # df -> dict
                items.append(df.to_dict())
        else:
            raise ValueError("annIds should be given as `dict` or `list` type.")

        return items

    def loadCats(self, catIds: Union[int, List[int]]) -> List[Dict[str, str]]:
        """カテゴリの `name` と `id` をキーに持つ辞書を list 形式で返す関数．

        Args:
            catIds (Union[int, List[int]]): 読み出したいカテゴリの `id`

        Raises:
            ValueError: catIds should be given as `int` or `list` type

        Returns:
            List[Dict[str, str]]: カテゴリの `name` と `id` をキーに持つ辞書．
            {
                "name": str,
                "id": int
            }
        """
        if isinstance(catIds, int):
            for i in range(len(ClassLabel)):
                if ClassLabel[i]["id"] == catIds:
                    items = [ClassLabel[i]]
        elif isinstance(catIds, list):
            items = []
            for k in catIds:
                for i in range(len(ClassLabel)):
                    if ClassLabel[i]["id"] == k:
                        items.append(ClassLabel[i])
        else:
            raise ValueError("catIds should be given as `int` or `list` type.")
        return items

    def loadImgs(self, imgIds: List[int]) -> List[Dict[str, str]]:
        """

        Args:
            imgIds (List[int]): [description]
        """
        # pandas.DataFrameの行を条件で抽出するquery
        # REF: https://note.nkmk.me/python-pandas-query/
        df = self.df.query(f"id in {imgIds}")

        df = df[["file_name", "id"]]
        # DataFrame -> Dict
        # REF: https://note.nkmk.me/python-pandas-to-dict/
        df = df.to_dict(orient="records")
        return df

    def get_df(self):
        return self.df


def getClassName(classID: int, cats: dict):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


def FilterARCDataset(
    data_root,
    cls_names: Union[List[str], None] = None,
    split: Literal["train", "val", "test"] = "train",
):
    """フィルタリングしたクラスのオブジェクトが映る画像をすべて読みだす関数．

    Args:
        data_root (str): データセットの root ディレクトリ．
        cls_names (Union(List[str], None), optional): 抽出するクラス名のリスト. Defaults to None.
        split (Literal["train", "val", "test"], optional): 読みだすデータセットの種類（'train' or 'val', or 'test'）. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    # initialize COCO api for instance annotations
    annFile = "{}/{}/annotations/instances_{}2017.json".format(data_root, split, split)
    coco = ARCDataset(annFile)

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

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


def getARCNormalMask(imgObj, cls_names, coco, catIds, ann_dir, input_img_size):
    annIds = coco.getAnnIds([imgObj["id"]])
    anns = coco.loadAnns(annIds)

    cats = coco.loadCats(catIds)
    mask = np.zeros((input_img_size["h"], input_img_size["w"]))  # mask [H, W]
    for a in range(len(anns)):
        className = getClassName(anns[a]["category_id"], cats)
        pixel_value = cls_names.index(className) + 1
        pth = anns[a]["anno_file_name"]
        pth = os.path.join(ann_dir, pth)
        new_mask = cv2.imread(pth, flags=0)
        if np.max(new_mask) > 1:
            raise ValueError("値が不正です．")
        new_mask = cv2.resize(
            new_mask * pixel_value,
            (input_img_size["w"], input_img_size["h"]),
        )
        mask = mask + new_mask

    return mask


def getARCBinaryMask(imgObj, coco, catIds, input_img_size) -> np.ndarray:
    annIds = coco.getAnnIds(imgObj["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)  # アノテーションを読みだす

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
    import pprint
    import sys

    sys.path.append("../../")
    from lib.config.config import pth

    root = os.path.join(pth.DATA_DIR, "ARCdataset")
    json_fp = os.path.join(
        pth.DATA_DIR,
        "ARCdataset",
        "train",
        "annotations",
        "instances_train2017.json",
    )

    ds = ARCDatasetTransformer(root, split="test")
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
    # pprint.pprint(df)

    # catNms = "item1"
    catNms = ["item1", "item2"]
    catIds = dataset.getCatIds(catNms)
    # print(catIds)

    imgIds = dataset.getImgIds(catIds)
    # print(imgIds)

    img_info = dataset.loadImgs(imgIds)
    # print(img_info)

    mask_info = dataset.getAnnIds(imgIds)
    print(mask_info)
