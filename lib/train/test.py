import os
import sys

sys.path.append("../..")
sys.path.append("../../../")

import torch
from yacs.config import CfgNode

from lib.config.config import pth
from lib.datasets.make_datasets import make_data_loader
from lib.models.make_network import make_network
from lib.train.trainers.make_trainer import make_trainer
from lib.utils.net_utils import load_network


def test(cfg: CfgNode):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = make_network(cfg).to(device)
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(
        network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch
    )
    trainer.val(epoch, val_loader, evaluator)


def main(cfg):
    # データの保存先を設定
    cfg.model_dir = os.path.join(
        pth.DATA_DIR, "result", cfg.task, cfg.model, cfg.model_dir
    )
