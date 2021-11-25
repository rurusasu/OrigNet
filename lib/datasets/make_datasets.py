import sys
from typing import Dict, List, Literal, Type, Union
from torch._C import TensorType

sys.path.append(".")
sys.path.append("../../")

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from yacs.config import CfgNode

from lib.datasets.dataset_catalog import DatasetCatalog
from lib.datasets.samplers import ImageSizeBatchSampler, IterationBasedBatchSampler
from lib.datasets.tasks.classify import ClassifyDataset
from lib.datasets.tasks.semantic_segm import SegmentationDataset
from lib.datasets.transforms import make_transforms


_dataset_factory = {"classify": ClassifyDataset, "semantic_segm": SegmentationDataset}


def make_dataset(
    # cfg: CfgNode,
    dataset_name: str,
    cls_names: Union[List[str], None] = None,
    task: Literal["classify", "semantic_segm"] = "classify",
    img_shape: Dict[str, int] = {"width": 224, "height": 224},
    mask_type: Literal["binary", "normal"] = "normal",
    transforms: Type[Union[None, TensorType]] = None,
):
    """
    `DatasetCatalog` から `dataset_name` を参照し，そこに保存されている情報をもとにデータセットを作成する関数

    Args:
        dataset_name (str): ロードするデータセット名．
        task (Literal["classify", "semantic_segm"], optional):
            適応させるタスク．
            Default to "classify".
        img_shape (Dict[str, int], optionanl):
            出力される画像のサイズ．
            Default to {"width": 224, "height": 224}.
        mask_type (Literal["binary", "normal"], optional):
                "binary": すべてのオブジェクトを単一のクラスとしてマスクする．
                "normal": オブジェクトをクラスごとにマスクする.
                Defaults to "normal".
        transforms (Type[Union[None, TensorType]]):
            データ拡張に使用するtorchvisionのクラス．
            Default to None.
    """
    if task != "classify" and task != "semantic_segm":
        raise ValueError("Invalid input for task.")

    args = DatasetCatalog.get(dataset_name)
    dataset = _dataset_factory[task]
    args["cls_names"] = cls_names
    args["img_shape"] = img_shape
    args["mask_type"] = mask_type
    # args["cfg"] = cfg
    del args["id"]

    # args["data_root"] = os.path.join(pth.DATA_DIR, args["data_root"])
    if transforms is not None:
        args["transforms"] = transforms

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
    # cfg: CfgNode,
    dataset_name: str,
    batch_size: int = 16,
    batch_sampler: Union[None, Literal["image_size"]] = None,
    ds_category: Literal["train", "val", "test"] = "train",
    img_shape: Dict[str, int] = {"width": 224, "height": 224},
    is_distributed: bool = False,
    max_iter: int = -1,
    normalization: bool = False,
    num_workers: int = 2,
    task: Literal["classify", "semantic_segm"] = "classify",
    toTensor: bool = True,
) -> torch.utils.data.DataLoader:
    """
    データローダーを作成する関数．

    Args:
        dataset_name (str): ロードするデータセット名．
        batch_size (int, optional): 1回の出力で読み出されるデータ数．
        batch_sampler (Union[None, Literal["image_size"]], optional): データサンプリング用のプログラム．
        default to None.
        ds_category (Literal["train", "val", "test"], optional): 使用するデータセットのカテゴリ名．
        defaults to "train".
        img_shape (Dict[str, int], optionanl): 出力される画像のサイズ．
        default to {"width": 224, "height": 224}.
        is_distributed (bool, optional): データをシャッフルしたものをテストに使用するか．
        defaults to False.
        max_iter (int, optional): イテレーションの最大値. defaults to -1.
        normalization (bool, optional): データを正規化する．default to Fales.
        num_workers (int, optional): 使用する `cpu` のワーカー数．default to 2.
        task (Literal["classify", "semantic_segm"], optional): 適応させるタスク．
        default to "classify".
        toTensor (bool, optional): torch.Tensor で出力する．
        `False` の場合，`ndarray` で出力．default to True.

    Returns:
        torch.utils.data.DataLoader: [description]
    """
    # --------------------------------------- #
    # 訓練データセットの場合のコンフィグ #
    # --------------------------------------- #
    if ds_category == "train":
        drop_last = False
        shuffle = True
    # --------------------------------------- #
    # 検証データセットの場合のコンフィグ #
    # --------------------------------------- #
    elif ds_category == "val":
        drop_last = False
        shuffle = True if is_distributed else False
    # ----------------------------------------- #
    # テストデータセットの場合のコンフィグ #
    # ----------------------------------------- #
    elif ds_category == "test":
        drop_last = False
        shuffle = True if is_distributed else False
    else:
        raise ("The required parameter for `make_data_loader` has not been set.")

    # dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    transforms = make_transforms(
        ds_category, toTensor=toTensor, normalization=normalization
    )
    # dataset = make_dataset(cfg, dataset_name=dataset_name, transforms=transforms)
    dataset = make_dataset(
        dataset_name=dataset_name, task=task, img_shape=img_shape, transforms=transforms
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


if __name__ == "__main__":
    import sys

    sys.path.append("../../")

    # from lib.config.config import cfg
    # from datasets.tasks.classify import Dataset

    cfg = CfgNode()
    cfg.train = CfgNode()
    cfg.task = "classify"
    cfg.train.dataset = "SampleTrain"
    cfg.train.batch_size = 4
    cfg.train.batch_sampler = ""
    cfg.train.num_workers = 2

    dataloader = make_data_loader(cfg)
    print(dataloader)

    # import torchvision
    # args = DatasetCatalog.get(cfg.train.dataset)
    # data_root = os.path.join(pth.DATA_DIR, args["data_root"])
    # dataset = torchvision.datasets.ImageFolder(data_root)
    # print(dataset)
    """
    class_to_idx: {'NG': 0, 'OK': 1}
    classes: ['NG', 'OK']
    imgs: [
        (img_path, 0),
        (img_path, 0), ...,
        (img_path, 1),
        (img_path, 1), ...
    ]
    targets: [
        000: 0,
        001: 0, ...
        050: 1,
        051: 1, ....
        ]
    """
