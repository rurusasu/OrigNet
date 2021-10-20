import os
import sys

sys.path.append("../..")
sys.path.append("../../../")

import torch
from yacs.config import CfgNode

from lib.config.config import pth
from lib.datasets.make_datasets import make_data_loader
from lib.evaluators.make_evaluator import make_evaluator
from lib.models.make_network import make_network
from lib.train.trainers.make_trainer import make_trainer
from lib.utils.net_utils import load_network


def test(cfg: CfgNode):
    # 検証用のデータローダーを作成
    val_loader = make_data_loader(cfg, is_train=False)

    # セマンティックセグメンテーションの場合，背景のクラスを追加しないと cross_entropy の計算でエラーが発生．
    # 理由は，画素値が 0 のラベルを与える必要があるため．
    cfg.num_classes = len(val_loader.dataset.cls_names) + 1
    network = make_network(cfg)

    trainer = make_trainer(cfg, network, device_name="auto")
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir)
    trainer.val(epoch, val_loader, evaluator)


def main(cfg):
    test(cfg)


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.cls_names = ["laptop", "tv"]
    cfg.task = "semantic_segm"
    cfg.network = "smp"
    cfg.model = "unetpp"
    cfg.encoder_name = "resnet18"
    cfg.model_dir = "model"
    cfg.train_type = "transfer"  # or scratch
    # cfg.train_type = "scratch"
    cfg.img_width = 224
    cfg.img_height = 224
    cfg.resume = True  # 追加学習するか
    cfg.record_dir = "record"
    cfg.ep_iter = -1
    cfg.skip_eval = False
    cfg.train = CfgNode()
    cfg.train.dataset = "COCO2017Val"
    cfg.train.criterion = ""
    cfg.train.metrics = "iou"
    cfg.use_amp = False
    cfg.test = CfgNode()
    # cfg.test.dataset = "SampleTest"
    # cfg.test.dataset = "BrakeRotorsTest"
    cfg.test.dataset = "COCO2017Val"
    cfg.test.batch_size = 20
    cfg.test.num_workers = 4
    # cfg.test.batch_sampler = "image_size"
    cfg.test.batch_sampler = ""

    # データの保存先を設定
    cfg.model_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.train.dataset, cfg.model, cfg.model_dir
    )
    cfg.record_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.test.dataset, cfg.model, cfg.record_dir
    )
    cfg.result_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.test.dataset, cfg.model, "result"
    )

    main(cfg)
