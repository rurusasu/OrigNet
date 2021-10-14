import os
import sys

sys.path.append(".")
sys.path.append("../../")
sys.path.append("../../../")

import torch
from yacs.config import CfgNode

from lib.config.config import pth, cfg
from lib.datasets.make_datasets import make_data_loader
from lib.models.make_network import make_network
from lib.train.scheduler import make_lr_scheduler
from lib.train.optimizers import make_optimizer
from lib.train.trainers.make_trainer import make_trainer
from lib.train.recorder import make_recorder
from lib.utils.net_utils import load_model, save_model


def train(cfg: CfgNode) -> None:
    """
    訓練と検証を行い，任意のエポック数ごとに訓練されたネットワークを保存する関数．

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
    """
    if "train" not in cfg:
        raise ("The training configuration is not set.")

    # cuda が存在する場合，cudaを使用する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PyTorchが自動で、処理速度の観点でハードウェアに適したアルゴリズムを選択してくれます。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    # 訓練と検証用のデータローダーを作成
    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)

    cfg.num_classes = len(train_loader.dataset.cls_names)
    # 指定した device 上でネットワークを生成
    network = make_network(cfg)
    trainer = make_trainer(
        cfg,
        network,
        device_name="auto",
    )
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    begin_epoch = load_model(
        network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume
    )

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        # 訓練途中のモデルを保存する
        if (epoch + 1) % cfg.save_ep == 0:
            save_model(
                network, optimizer, scheduler, recorder, epoch + 1, cfg.model_dir
            )

        # 検証
        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, recorder=recorder)

    # 訓練終了後のモデルを保存
    save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)
    # trainer.val(epoch, val_loader, evaluator, recorder)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    return network


def main(cfg):
    cfg.model_dir = os.path.join(
        pth.DATA_DIR,
        "trained",
        cfg.model_dir,
    )
    cfg.record_dir = os.path.join(
        pth.DATA_DIR,
        "trained",
        cfg.record_dir,
    )
    # 訓練
    train(cfg)


if __name__ == "__main__":
    # テスト
    import traceback

    debug = True

    if debug:
        from yacs.config import CfgNode as CN

        """
        cfg = CN()
        cfg.task = "classify"
        cfg.network = "cnns"
        cfg.model = "res_18"
        cfg.cls_names = ["laptop", "tv"]
        cfg.encoder_name = "resnet18"
        cfg.model_dir = "model"
        cfg.train_type = "transfer"  # or scratch
        # cfg.train_type = "scratch"
        cfg.img_width = 224
        cfg.img_height = 224
        cfg.resume = True  # 追加学習するか
        cfg.record_dir = "record"
        cfg.ep_iter = -1
        cfg.save_ep = 5
        cfg.eval_ep = 1
        cfg.train = CN()
        cfg.train.epoch = 15
        cfg.train.dataset = "SampleTrain"
        cfg.train.batch_size = 4
        cfg.train.num_workers = 2
        cfg.train.batch_sampler = ""
        cfg.train.optim = "adam"
        cfg.train.criterion = ""
        cfg.train.lr = 1e-3
        cfg.train.scheduler = "step_lr"
        cfg.train.weight_decay = 0.0
        cfg.train.milestones = (20, 40, 60, 80, 100, 120, 160, 180, 200, 220)
        cfg.train.gamma = 0.5
        cfg.train.metrics = "iou"
        cfg.test = CN()
        cfg.test.dataset = "SampleTest"
        cfg.test.batch_size = 20
        cfg.test.num_workers = 2
        cfg.test.batch_sampler = ""
        """

        conf = CN()
        conf.cls_names = ["laptop", "tv"]
        conf.task = "semantic_segm"
        conf.network = "smp"
        conf.model = "unetpp"
        conf.encoder_name = "resnet18"
        conf.model_dir = "model"
        conf.record_dir = "record"
        conf.train_type = "transfer"  # or scratch
        # cfg.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.use_amp = False
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
        conf.train = CN()
        conf.train.epoch = 20
        # cfg.train.dataset = "SampleTrain"
        # cfg.train.dataset = "Sample_2Train"
        # cfg.train.dataset = "BrakeRotorsTrain"
        # cfg.train.dataset = "LinemodTrain"
        conf.train.dataset = "COCO2017Val"
        conf.train.batch_size = 20
        conf.train.num_workers = 2
        conf.train.batch_sampler = ""
        conf.train.optim = "adam"
        conf.train.criterion = ""
        conf.train.lr = 1e-3
        conf.train.scheduler = "step_lr"
        conf.train.weight_decay = 0.0
        conf.train.milestones = (20, 40, 60, 80, 100, 120, 160, 180, 200, 220)
        conf.train.warp_iter = 50
        conf.train.gamma = 0.5
        conf.train.metrics = "iou"
        conf.test = CN()
        # cfg.test.dataset = "SampleTest"
        # cfg.test.dataset = "Sample_2Test"
        # cfg.test.dataset = "LinemodTest"
        conf.test.dataset = "COCO2017Val"
        conf.test.batch_size = 20
        conf.test.num_workers = 2
        conf.test.batch_sampler = ""

        conf.model_dir = os.path.join(
            conf.task, conf.train.dataset, conf.model, conf.model_dir
        )
        conf.record_dir = os.path.join(
            conf.task, conf.train.dataset, conf.model, conf.record_dir
        )

        torch.cuda.empty_cache()
        try:
            main(conf)
        except:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    else:
        torch.cuda.empty_cache()
        try:
            main(cfg)
        except:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
