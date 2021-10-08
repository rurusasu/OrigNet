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


def train_prof(cfg: CfgNode) -> None:
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
    # 訓練と検証用のデータローダーを作成
    train_loader = make_data_loader(cfg, is_train=True, max_iter=cfg.ep_iter)

    cfg.num_classes = len(train_loader.dataset.cls_names)
    # 指定した device 上でネットワークを生成
    network = make_network(cfg)
    criterion = smp.utils.losses.DiceLoss()
    metrics = smp.utils.metrics.IoU()
    optimizer = torch.optim.Adam(
        network.parameters(), cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    use_amp = True
    scaler = amp.GradScaler(enabled=use_amp)

    network = network.to(device)
    network.train()

    if not os.path.exists(cfg.profile_dir):
        os.makedirs(cfg.profile_dir)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # schedule=torch.profiler.schedule(wait=1, warmup=14, active=3, repeat=2),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
    ) as prof:

        with tqdm.tqdm(total=len(train_loader), leave=False, desc="train") as pbar:
            for iteration, batch in enumerate(train_loader):
                # 混合精度テスト
                # optimizer の初期化
                optimizer.zero_grad()
                # 演算を混合精度でキャスト
                with amp.autocast(enabled=use_amp):
                    # もし，混合精度を使用する場合．
                    if use_amp:
                        input = batch["img"].to(
                            device=device, dtype=torch.float16, non_blocking=True
                        )
                        target = batch["target"].to(
                            device=device, dtype=torch.float16, non_blocking=True
                        )
                    # 混合精度を使用しない場合
                    else:
                        input = batch["img"].to(device=device, non_blocking=True)
                        target = batch["target"].to(device=device, non_blocking=True)

                    output = network(input)

                    loss = criterion(output, target)
                    iou = metrics(output, target)

                # 損失をスケーリングし、backward()を呼び出してスケーリングされた微分を作成する
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.update()
                prof.step()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def main_prof(cfg):
    cfg.profile_dir = os.path.join(
        pth.DATA_DIR, "trained", cfg.task, cfg.train.dataset, cfg.model, cfg.profile_dir
    )
    # 訓練
    train_prof(cfg)


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
