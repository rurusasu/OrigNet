import json
import os
import yaml
from glob import glob
from typing import Any, Dict, List, Union

import cv2
import ndjson
import numpy as np
import skimage.io as io
import torch
from tqdm.contrib import tenumerate
from yacs.config import CfgNode


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


def CfgSave(config: CfgNode, save_dir: str) -> Any:
    """CfgNode を yaml ファイルとして保存するための関数．

    Args:
        config (CfgNode): 訓練の条件設定が保存された辞書．
        save_dir (str): CfgNode を保存するための yaml ファイルのパス．
    """
    dic = {}
    w_pth = os.path.join(save_dir, f"{config.task}_{config.model}.yaml")
    # CfgNode -> Dict
    # この変換をしない場合，不要な変数が YAML に保存される．
    for k, v in config.items():
        dic[k] = v

    with open(w_pth, "w") as yf:
        yaml.dump(dic, yf, default_flow_style=False)


def DirCheckAndMake(dir_pth: str) -> str:
    """
    dir_pth の先のディレクトリが存在するかを判定し，存在しなければ作成する関数．

    Arg:
        dir_pth (str): ディレクトリのパス．

    Return:
        dir_pth (str): ディレクトリのパス．
    """
    if not os.path.exists(dir_pth):
        os.makedirs(dir_pth)
    return dir_pth


def GetImgFpsAndLabels(
    data_root: str,
    cls_names: Union[List[str], None] = None,
):
    """指定したディレクトリ内の画像ファイルパス(Image file paths)とクラスラベルの一覧を取得する関数．

    Arg:
        data_root (str): 画像データが格納された親フォルダ
        cls_names (Union[List[str], None], optional):
                読みだしたいクラス名のリスト.
                `None` の場合，すべてのクラスを読みだす．
                Defaults to None.
    Return:
        classes (list): クラス名のリスト.
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
    for i, p in tenumerate(glob(os.path.join(data_root, "*"))):
        cls_name = os.path.basename(p.rstrip(os.sep))
        if cls_names is not None:
            if cls_name not in cls_names:
                continue

        # クラス名のリストを作成
        classes.append(cls_name)
        # クラス名と label_num を対応させる辞書を作成
        class_to_idx[cls_name] = i

        if os.path.exists(os.path.join(p, "imgs")) and os.path.exists(
            os.path.join(p, "masks")
        ):
            # RGB 画像を探索
            for img_pth in glob(os.path.join(p, "imgs", "**"), recursive=True):
                if os.path.isfile(img_pth) and os.path.splitext(img_pth)[1] in file_ext:
                    # 画像データパスをリストに格納
                    imgs.append(img_pth)
                    # label_num のみ格納したリストを作成
                    targets.append(i)

            # mask 画像を探索
            for msk_pth in glob(os.path.join(p, "masks", "**"), recursive=True):
                if os.path.isfile(msk_pth) and os.path.splitext(msk_pth)[1] in file_ext:
                    # マスク画像データパスをリストに格納
                    msks.append(msk_pth)

        # クラス名ディレクトリ内の file_ext にヒットするパスを全探索
        else:
            # ディレクトリ内のすべての画像を読みだす．
            for img_pth in glob(os.path.join(p, "**"), recursive=True):
                if os.path.isfile(img_pth) and os.path.splitext(img_pth)[1] in file_ext:
                    # データパスと label_num を格納したタプルを作成
                    imgs.append(img_pth)
                    # label_num のみ格納したリストを作成
                    targets.append(i)

    return classes, class_to_idx, imgs, targets, msks


def LoadImgAndResize(img_fp: str, input_img_size: Dict[int, int]) -> np.ndarray:
    """
    画像パスのリストから、id で指定された画像を読みだす関数

    Args:
        img_fp (str): 画像ファイルパス
        input_img_size (Tuple[int, int]): 画像のサイズ．
        例: input_img_size{ "w" : 255, "h" : 255 }

    Returns:
        np.ndarray: `[H, W, 3]` の ndarray
    """
    # Read and normalize an image
    # ndarray([H, W, C])
    img = io.imread(img_fp) / 255.0
    # Resize: [H, W, C] -> [H', W', C]
    # 変数が，[W, H] で与える点に注意
    img = cv2.resize(img, (input_img_size["w"], input_img_size["h"]))
    if len(img.shape) == 3 and img.shape[2] == 3:  # If it is a RGB 3 channel image
        return img
    else:  # 白黒の画像を扱う場合は、次元を3にする
        stacked_img = np.stack((img,) * 3, axis=-1)
        return stacked_img


def LoadNdjson(json_pth: str) -> Dict:
    """
    WriteDataToNdjson を使って保存した JSON ファイルからデータを読みだす関数．

    Args:
        json_pth (str): 読み出す json ファイル．

    Return:
        data (Dict): json ファイルに保存されていたデータ．
    """
    fp = json_pth
    if ".json" not in os.path.splitext(fp)[-1]:
        fp = json_pth + ".json"
    with open(fp) as f:
        data = ndjson.load(f)

    return data


def OneTrainDir(root_dir: str = ".", dir_name: str = "default") -> str:
    """
    1回の訓練の全データを保存するディレクトリを作成する関数．
    ディレクトリ名は，cfg.task で与えられる．

    Args:
        root_dir (str, optional): 親ディレクトリのパス．Default to ".".
        dir_name (str, optional): 作成するディレクトリ名．Default to "default"．

    """
    dir_pth = os.path.join(root_dir, dir_name)
    dir_pth = DirCheckAndMake(dir_pth)
    return dir_pth


def OneTrainLogDir(config: CfgNode, root_dir: str = ".") -> List[str]:
    """
    1回の訓練と検証時のデータを保存する`model`，`record`，`result` ディレクトリを作成するための関数．

    Args:
        config (CfgNode): 訓練の条件設定が保存された辞書．
        root_dir (str): 親ディレクトリのパス．
    """

    model_dir = DirCheckAndMake(os.path.join(root_dir, config.model_dir))
    record_dir = DirCheckAndMake(os.path.join(root_dir, config.record_dir))
    result_dir = DirCheckAndMake(os.path.join(root_dir, config.result_dir))

    return [model_dir, record_dir, result_dir]


def SelectDevice(max_gpu_num: int = 0):
    """
    CUDA GPU が使用できるかを判定し，使用できればその個数を取得，できなければ cpu を選択する関数．
    GPU 情報の取得 および 個数のカウント方法 は以下のサイトを参照．
    REF: https://note.nkmk.me/python-pytorch-cuda-is-available-device-count/
    Arg:
        max_gpu_num(int): 使用する GPU の最大個数．0 <= n <= max_gpu_count で指定する．Default to 0.

    Returns:
        device_name(str): 使用可能なデバイス名 ("cpu" or "cuda")．
        num_devices(List[int]): 使用可能なデバイスの番号．
        `device_name="cpu"` : `num_devices=[]`．
        GPUが1つしか搭載されていない場合，`device_name="cuda"` : `num_devices=[0]`．
        GPUが複数搭載されている場合，`device_name="cuda"` : `num_devices=[0, 1, ...]`.

    """
    if torch.cuda.is_available():  # GPU が使用可能な場合
        num_devices = torch.cuda.device_count()
        if num_devices == 1:  # GPU が1つしか搭載されていない場合
            return "cuda", [0]
        else:  # GPU が 2 つ以上搭載されている場合
            gpu_num = []
            for i in range(num_devices):
                gpu_num.append(i)
                if num_devices < max_gpu_num:
                    break
            return "cuda", gpu_num
    else:  # GPU が使用不可
        return "cpu", []


def Tensor2Ndarray3D(x: torch.Tensor) -> np.ndarray:
    """
    画像としての形を保ったまま `torch.Tensor` から `ndarray` に変換する関数．

    Args:
        x (torch.Tensor): 画像などの3次元配列 ([C, H, W])．

    Returns:
        np.ndarray: 画像などの3次元配列 ([H, W, C]).
    """
    x = x.detach().clone().cpu()
    # Tensor[C, H, W] -> ndarray[H, W, C]
    return x.numpy().transpose(1, 2, 0)


def WriteDataToJson(data: Dict, wt_json_pth: str):
    """
    データを JSON ファイルに出力する関数．

    Args:
        data(Dict): json ファイルに出力するデータ．
        wt_json_pth (str): データを書き込む `JSON` ファイルへのパス．
    """
    # Dict -> Json
    json_data = json.dumps(data)

    # path に .json が含まれていなければ追加
    fp = wt_json_pth
    if ".json" not in os.path.splitext(fp)[-1]:
        fp = fp + ".json"
    # ファイルへの書き込み
    with open(fp, "wt") as f:
        json.dump(json_data, f, ensure_ascii=True)


def WriteDataToNdjson(data: Dict, wt_json_pth: str):
    """
    データを追記する形で JSON ファイルに出力する関数．
    json は，保存したい全てのデータを一度読みだしておく必要があるが ndjson は，1つ1つのデータを順次保存することができる．
    ndjson の使い方については以下を参照．
    REF: https://qiita.com/eg_i_eg/items/aff02f6057b476cb15fa

    Args:
        data (Dict): json ファイルに出力するデータ．
        wt_json_pth (str): データを書き込む `JSON` ファイルへのパス．
    """
    # path に .json が含まれていなければ追加
    fp = wt_json_pth
    if ".json" not in os.path.splitext(fp)[-1]:
        fp = fp + ".json"
    # ファイルへの書き込み
    with open(fp, "a") as f:
        writer = ndjson.writer(f)
        writer.writerow(data)


if __name__ == "__main__":
    print(SelectDevice())
