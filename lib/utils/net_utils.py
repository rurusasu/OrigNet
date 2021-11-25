import gc
import os
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp


def load_model(
    network,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    recorder,
    model_dir: str,
    resume: bool = True,
    epoch: int = -1,
):
    """
    事前学習により保存されたモデルを読みだす関数

    Args:
        network()
    """
    if not resume:
        os.system("rm -rf {}".format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print("Load model: {}".format(os.path.join(model_dir, "{}.pth".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    network.load_state_dict(pretrained_model["net"])
    optimizer.load_state_dict(pretrained_model["optim"])
    scheduler.load_state_dict(pretrained_model["scheduler"])
    recorder.load_state_dict(pretrained_model["recorder"])
    return pretrained_model["epoch"] + 1


def load_network(
    network: torch.nn,
    model_dir: str,
    resume: bool = True,
    epoch: int = -1,
    strict: bool = True,
) -> Union[torch.nn.Module, int]:
    """保存されたネットワークのパラメタを読みだす関数．

    Args:
        network (torch.nn): パラメタを代入するためのネットワーク構造
        model_dir (str): パラメタの読み出し先．
        resume (bool, optional): 追加学習のする/しない．False の場合 0 を返す. Defaults to True.
        epoch (int, optional): 読みだすパラメタの訓練回数. Defaults to -1.
        strict (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pth" in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print("Load model: {}".format(os.path.join(model_dir, "{}.pth".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    network.load_state_dict(pretrained_model["net"], strict=strict)

    return pretrained_model["epoch"] + 1


def save_model(
    network,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    recorder,
    epoch: int,
    model_dir: str,
):
    """
    訓練されたモデルを保存する関数
    """
    os.system("mkdir -p {}".format(model_dir))
    torch.save(
        {
            "net": network.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "recorder": recorder.state_dict(),
            "epoch": epoch,
        },
        os.path.join(model_dir, "{}.pth".format(epoch)),
    )

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:
        return
    os.system("rm {}".format(os.path.join(model_dir, "{}.pth".format(min(pths)))))


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size())
            )
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    # BatchNormのみFP32フォーマットにしないと性能が出ない。
    # BatchNormレイヤを検索し、このレイヤのみFP32に設定。
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    ネットワークのパラメタを半精度に変換する関数．
    以下のサイトからコピー
    REF: https://aru47.hatenablog.com/entry/2020/11/06/225942
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


def _log_api_usage_once(obj: str) -> None:  # type: ignore
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return
    # NOTE: obj can be an object as well, but mocking it here to be
    # only a string to appease torchscript
    if isinstance(obj, str):
        torch._C._log_api_usage_once(obj)
    else:
        torch._C._log_api_usage_once(f"{obj.__module__}.{obj.__class__.__name__}")


def reduce_loss_stats(loss_stats: dict) -> dict:
    """
    損失の統計情報を平均化する関数
    """
    reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
    return reduced_losses


def _TrainingPreProcessing():
    """
    訓練の前処理用関数

    参考: [amp_recipe.ipynb](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/19350f3746283d2a6e32a8e698f92dc4/amp_recipe.ipynb#scrollTo=pCpWeg5PF-dw)
    """
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()


def _TrainingPostProcessing():
    """
    訓練の後処理用関数

    参考: [amp_recipe.ipynb](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/19350f3746283d2a6e32a8e698f92dc4/amp_recipe.ipynb#scrollTo=pCpWeg5PF-dw)
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def train(
    network: torch.nn,
    epoch: int,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    recorder,
    device: torch.device = "cpu",
    use_amp: bool = False,
    batch_multiplier: int = 1,
):
    """
    DataLoader から 1Epoch 分だけデータを取り出して訓練を実行する関数．

    Args:
        network (torch.nn): 訓練対象のネットワーク．
        epoch (int): エポック数．
        data_loader (torch.utils.data.DataLoader): データローダー
        optimizer (torch.optim): 最適化関数．
        recorder ([type]): Log 保存用のクラス．
        device (torch.device): 訓練に使用するハードウェア．Default to "cpu".
        use_amp (bool, optional): 半精度で訓練を行うか. Defaults to False.
        batch_multiplier (int, optional): 指定された batch 数の平均誤差を逆伝搬する.
    """
    # ---- amp setting ---- #
    # self.use_amp = use_amp
    scaler = amp.GradScaler(enabled=use_amp)
    network.train()
    max_iter = len(data_loader)
    if isinstance(batch_multiplier, int) and (batch_multiplier > 0):
        multiple_mumibach_count = batch_multiplier  # multiple minibatch
    else:
        raise ValueError("batch_multiplier は int 型の 0 より大きい値を設定してください。")

    _TrainingPreProcessing()
    with tqdm(total=len(data_loader), leave=True, desc="train") as pbar:
        for iteration, batch in enumerate(data_loader):
            # optimizer の初期化
            optimizer.zero_grad()
            t_iter_start = time.time()
            iteration += 1
            recorder.step += 1

            # --------------- #
            # training stage #
            # --------------- #
            # use_amp = True の場合，混合精度を用いて訓練する．
            # 演算を混合精度でキャスト
            with amp.autocast(enabled=use_amp):
                input = batch["img"].cuda(device)
                target = batch["target"].cuda(device)
                if use_amp:  # もし，混合精度を使用する場合．
                    input = input.to(torch.half)
                    target = target.to(torch.half)

                _, loss, loss_stats, image_stats = network(input, target)

                loss = loss / batch_multiplier

            # 損失をスケーリングし、backward()を呼び出してスケーリングされた微分を作成する
            scaler.scale(loss).backward()
            # グラデーションのスケールを解除し、optimizer.step()を呼び出すかスキップする。
            if multiple_mumibach_count == 0:
                scaler.step(optimizer)
                scaler.update()
                multiple_mumibach_count = batch_multiplier

            multiple_mumibach_count -= 1  # multiple minibatch

            batch_time = time.time() - t_iter_start
            recorder.batch_time.update(batch_time)
            recorder.update_loss_stats(loss_stats)
            pbar.update()

            if iteration % 10 == 0 or iteration == (max_iter - 1):
                """
                lr = optimizer.param_groups[0]["lr"]
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                training_state = "  ".join(
                    ["{}", "lr: {:.6f}", "max_memory: {:.0f} MB"]
                )
                training_state = training_state.format(
                    # eta_string,
                    str(recorder),
                    lr,
                    memory,
                )
                # print(training_state)
                """
                # record loss_stats and image_dict
                if image_stats:
                    recorder.update_image_stats(image_stats)

                recorder.record("train")

            # 【PyTorch】不要になった計算グラフを削除してメモリを節約
            # REF: https://tma15.github.io/blog/2020/08/22/pytorch%E4%B8%8D%E8%A6%81%E3%81%AB%E3%81%AA%E3%81%A3%E3%81%9F%E8%A8%88%E7%AE%97%E3%82%B0%E3%83%A9%E3%83%95%E3%82%92%E5%89%8A%E9%99%A4%E3%81%97%E3%81%A6%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84/
            # 誤差逆伝播を実行後、計算グラフを削除
            del input, target, loss, loss_stats, image_stats
            gc.collect()
            torch.cuda.empty_cache()

    _TrainingPostProcessing()


def val(
    network: torch.nn,
    epoch: int,
    data_loader: torch.utils.data.DataLoader,
    evaluator=None,
    recorder=None,
    device: torch.device = "cpu",
    use_amp: bool = False,
) -> np.ndarray:
    """
    1 Epoch 分のデータを用いて検証を行う関数．

    Args:
        network (torch.nn): 訓練対象のネットワーク．
        epoch (int): エポック数．
        data_loader (torch.utils.data.DataLoader): データローダー
        evaluator ([type], optional): 特殊な検証の場合に使用するクラス. Defaults to None.
        recorder ([type], optional): Log 保存用のクラス．Defaults to None.
        device (torch.device, optional): 訓練に使用するハードウェア．Defaults to "cpu".
        use_amp (bool, optional): 半精度で訓練を行うか. Defaults to False.

    Returns:
        [type]: [description]
    """
    # 検証の前処理
    _TrainingPostProcessing()
    network.eval()
    val_loss_stats = {}
    data_size = len(data_loader)
    with tqdm(total=len(data_loader), leave=True, desc="val") as pbar:
        for iteration, batch in enumerate(data_loader):
            with torch.no_grad():
                with amp.autocast(enabled=use_amp):
                    # non_blocking については以下を参照
                    # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
                    # 【PyTorch】地味に知っておくべき実装の躓きドコロ
                    # REF: https://www.hellocybernetics.tech/entry/2018/02/20/182906
                    input = batch["img"].cuda(device)
                    target = batch["target"].cuda(device)

                    output, _, loss_stats, image_stats = network(input, target)

                if evaluator is not None:
                    result = evaluator.evaluate(
                        iteration=iteration, batch_output=output, batch=batch
                    )
                pbar.update()

            loss_stats = reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

    loss_state = []
    for k in val_loss_stats.keys():
        val_loss_stats[k] /= data_size
        loss_state.append("{}: {:.4f}".format(k, val_loss_stats[k]))
    print(loss_state)

    if evaluator is not None:
        result = evaluator.summarize()
        val_loss_stats.update(result)

    if recorder:
        recorder.record("val", epoch, val_loss_stats, image_stats)

    del input, target, output, image_stats
    gc.collect()
    torch.cuda.empty_cache()
    # 検証の後処理
    _TrainingPostProcessing()

    return val_loss_stats["batch_loss"].cpu().detach().clone().numpy()
