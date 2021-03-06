import os
import sys

sys.path.append("../..")
sys.path.append("../../../")

import numpy as np
import torch
from yacs.config import CfgNode

from lib.config.config import pth, cfg
from lib.datasets.make_datasets import make_data_loader
from lib.evaluators.make_evaluator import make_evaluator
from lib.models.make_network import make_network
from lib.train.trainers.make_trainer import make_trainer
from lib.utils.net_utils import load_network


def test(cfg: CfgNode) -> np.ndarray:
    # 検証用のデータローダーを作成
    val_loader = make_data_loader(
        dataset_name=cfg.test.dataset,
        batch_size=cfg.test.batch_size,
        batch_sampler=cfg.test.batch_sampler,
        ds_category="test",
        img_shape={"width": cfg.img_width, "height": cfg.img_height},
        num_workers=cfg.test.num_workers,
        task=cfg.task,
    )

    # セマンティックセグメンテーションの場合，背景のクラスを追加しないと cross_entropy の計算でエラーが発生．
    if cfg.task == "classify":
        cfg.num_classes = len(val_loader.dataset.cls_names)
    elif cfg.task == "semantic_segm":
        # 理由は，画素値が 0 のラベルを与える必要があるため．
        cfg.num_classes = len(val_loader.dataset.cls_names) + 1
    network = make_network(
        model_name=cfg.model,
        num_classes=cfg.num_classes,
        network_name=cfg.network,
        encoder_name=cfg.encoder_name,
        replaced_layer_num=cfg.replaced_layer_num,
        train_type=cfg.train_type,
    )

    trainer = make_trainer(cfg, network, device_name="auto")
    evaluator = make_evaluator(cfg, cls_names=val_loader.dataset.cls_names)
    epoch = load_network(network, cfg.model_dir)
    val_loss = trainer.val(epoch, val_loader, evaluator)

    # 不要なオブジェクトを削除
    del network, trainer, evaluator

    return val_loss


def main(cfg):
    test(cfg)


if __name__ == "__main__":
    import traceback

    debug = False
    torch.cuda.empty_cache()
    if not debug:
        try:
            main(cfg)
        except Exception as e:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    else:
        print("テストをデバッグモードで実行します．")

        conf = CfgNode()
        conf.task = "classify"
        conf.network = "cnns"
        conf.model = "res_18"
        conf.model_dir = "model"
        conf.train_type = "transfer"  # or scratch
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.use_amp = False  # 半精度で訓練するか
        conf.record_dir = "record"
        conf.ep_iter = -1
        # conf.skip_eval = False
        conf.train = CfgNode()
        conf.train.dataset = "SampleTrain"
        # conf.train.dataset = "AngleDetectTrain_2"
        conf.train.criterion = ""
        conf.test = CfgNode()
        conf.test.dataset = "SampleTest"
        # conf.test.dataset = "BrakeRotorsTest"
        # conf.test.dataset = "AngleDetectTest"
        conf.test.batch_size = 20
        conf.test.num_workers = 4
        # conf.test.batch_sampler = "image_size"
        conf.test.batch_sampler = ""

        """
        conf = CfgNode()
        conf.cls_names = ["laptop", "tv"]
        conf.task = "semantic_segm"
        conf.network = "smp"
        conf.model = "unetpp"
        conf.encoder_name = "resnet18"
        conf.model_dir = "model"
        conf.train_type = "transfer"  # or scratch
        # conf.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.record_dir = "record"
        conf.ep_iter = -1
        conf.skip_eval = False
        conf.train = CfgNode()
        conf.train.dataset = "COCO2017Val"
        conf.train.criterion = ""
        conf.train.metrics = "iou"
        conf.use_amp = False
        conf.test = CfgNode()
        # conf.test.dataset = "SampleTest"
        # conf.test.dataset = "BrakeRotorsTest"
        conf.test.dataset = "COCO2017Val"
        conf.test.batch_size = 20
        conf.test.num_workers = 4
        # conf.test.batch_sampler = "image_size"
        conf.test.batch_sampler = ""
        """

        # データの保存先を設定
        conf.model_dir = os.path.join(
            pth.DATA_DIR,
            "trained",
            conf.task,
            conf.train.dataset,
            conf.model,
            conf.model_dir,
        )
        conf.record_dir = os.path.join(
            pth.DATA_DIR,
            "trained",
            conf.task,
            conf.test.dataset,
            conf.model,
            conf.record_dir,
        )
        conf.result_dir = os.path.join(
            pth.DATA_DIR, "trained", conf.task, conf.test.dataset, conf.model, "result"
        )

        try:
            main(conf)
        except Exception as e:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
