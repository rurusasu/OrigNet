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
    # cuda が存在する場合，cudaを使用する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 検証用のデータローダーを作成
    val_loader = make_data_loader(cfg, is_train=False)

    cfg.num_classes = len(val_loader.dataset.cls_names)
    network = make_network(cfg).to(device)

    trainer = make_trainer(cfg, network, device)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network, cfg.model_dir)
    trainer.val(epoch, val_loader, evaluator)


def main(cfg):
    # データの保存先を設定
    cfg.result_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.model, "result"
    )
    cfg.record_dir = cfg.result_dir
    cfg.model_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.model, cfg.model_dir
    )
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
    cfg.train.criterion = ""
    cfg.train.metrics = "iou"
    cfg.test = CfgNode()
    # cfg.test.dataset = "SampleTest"
    # cfg.test.dataset = "BrakeRotorsTest"
    cfg.test.dataset = "COCO2017Val"
    cfg.test.batch_size = 20
    cfg.test.num_workers = 4
    # cfg.test.batch_sampler = "image_size"
    cfg.test.batch_sampler = ""

    main(cfg)
