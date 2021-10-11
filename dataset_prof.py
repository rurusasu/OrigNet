import os
import sys

from torch import profiler

sys.path.append(".")
sys.path.append("../../")
sys.path.append("../../../")

import segmentation_models_pytorch as smp
import torch
import tqdm
from torch.cuda import amp
from yacs.config import CfgNode

from lib.config.config import pth, cfg
from lib.datasets.make_datasets import make_data_loader
from lib.models.make_network import make_network


def dataset_prof(cfg: CfgNode) -> None:
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
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=3, active=13),
        on_trace_ready=profiler.tensorboard_trace_handler(cfg.profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with profiler.record_function("dataset"):
            # 訓練と検証用のデータローダーを作成
            train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)
            with tqdm.tqdm(total=len(train_loader), leave=False, desc="train") as pbar:
                for iteration, batch in enumerate(train_loader):
                    pbar.update()
                    prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


def main_prof(cfg):
    cfg.profile_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.train.dataset, cfg.model, cfg.profile_dir
    )
    # 訓練
    dataset_prof(cfg)


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
        conf.profile_dir = "profile"
        conf.train_type = "transfer"  # or scratch
        # cfg.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
        conf.train = CN()
        conf.train.epoch = 3
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

        torch.cuda.empty_cache()
        try:
            main_prof(conf)
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
