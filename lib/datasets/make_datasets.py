import importlib.machinery
import importlib.util
import os
import sys
import time
from typing import Type, Union
from torch._C import TensorType


sys.path.append(".")
sys.path.append("../../")

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from yacs.config import CfgNode

from . import samplers
from .dataset_catalog import DatasetCatalog
from .transforms import make_transforms
from lib.config.config import pth


def _dataset_factory(pth, task: str) -> object:
    """データセット名に合わせて作成されたディレクトリ内のファイルを読みだす関数

    Args:
        data_source (str): DataCatalog に保存されているデータセット名と同じ文字列
        task (str): 実行するタスク名．例: 'classify'

    Returns:
        object: 引数で指定した python ソースファイル内に記述されている関数
    """
    pth = os.path.join(pth.LIB_DIR, "datasets", "tasks", task + ".py")
    spec = importlib.util.spec_from_file_location(task, pth)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)

    return my_module.Dataset


def make_dataset(
    cfg: CfgNode,
    dataset_name: str,
    transforms: Type[Union[None, TensorType]] = None,
    is_train: bool = True,
):
    """`DatasetCatalog` から `dataset_name` を参照し，そこに保存されている情報をもとにデータセットを作成する関数
    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        dataset_name (str): ロードするデータセット名
        transforms (torchvision.transforms): データ拡張に使用するtorchvisionのクラス．default to None.
        is_train (bool): 訓練用データセットか否か．default to True.
    """
    args = DatasetCatalog.get(dataset_name)
    dataset = _dataset_factory(pth, cfg.task)
    args["cfg"] = cfg
    del args["id"]

    # args["data_root"] = os.path.join(pth.DATA_DIR, args["data_root"])
    if transforms is not None:
        args["transforms"] = transforms
    args["split"] = "train" if is_train else "test"
    dataset = dataset(**args)

    return dataset


def _make_data_sampler(dataset, shuffle: bool):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _make_batch_data_sampler(
    cfg: CfgNode,
    sampler: Sampler,
    batch_size: int,
    drop_last: bool,
    max_iter: int,
    is_train: bool = True,
):
    """イタレーションごとにデータセットからデータをサンプリングする際に行う処理を決定する関数

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        sampler (torch.utils.data.sampler.Sampler): データセットからデータをサンプリングする際の処理を自動化するクラス
        batch_size (int): バッチサイズ
        drop_last (bool): サンプリングしきれなかった余りを切り捨てるか
        max_iter (int): イテレーションの最大値
        is_train (bool): 訓練用データセットか否か．default to True.
    """
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last
    )
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)

    strategy = cfg.train.batch_sampler if is_train else cfg.test.batch_sampler
    if strategy == "image_size":
        batch_sampler = samplers.ImageSizeBatchSampler(
            sampler, batch_size, drop_last, 256, 480, 640
        )

    return batch_sampler


def _worker_init_fn(worker_id):
    """
    workerの初期化時に乱数のシードを個別に設定する関数
    これにより、workerの分だけforkするときに同一のnumpyのRandom Stateの状態がコピーされるため生じる入力データの重複を避けることができる。
    REF: https://qiita.com/kosuke1701/items/14cd376e024f86e57ff6
    """
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


def make_data_loader(
    cfg: CfgNode,
    is_train: bool = True,
    is_distributed: bool = False,
    max_iter: int = -1,
):
    """データローダーを作成する関数．

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        is_train (bool): 訓練用データセットか否か．default to True.
        is_distributed (bool): データをシャッフルしたものをテストに使用するか．default to False.
        max_iter (int): イテレーションの最大値．default to -1.
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

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(
        cfg, dataset_name=dataset_name, transforms=transforms, is_train=is_train
    )
    sampler = _make_data_sampler(dataset, shuffle)
    batch_sampler = _make_batch_data_sampler(
        cfg, sampler, batch_size, drop_last, max_iter, is_train
    )
    num_workers = cfg.train.num_workers
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
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
