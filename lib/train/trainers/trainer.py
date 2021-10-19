import datetime
import gc
import sys
import time
from typing import Literal

sys.path.append("../../../")

import torch
from tqdm import tqdm
from torch.cuda import amp
from torch.nn import DataParallel

from lib.utils.base_utils import SelectDevice


class Trainer(object):
    def __init__(
        self, network, device_name=Literal["cpu", "cuda", "auto"], use_amp: bool = True
    ):
        """
        device 引数について不明な場合は以下を参照．
        REF: https://note.nkmk.me/python-pytorch-device-to-cuda-cpu/

        Args:
            network: 訓練されるネットワーク
            device(str): 'cpu' もしくは 'cuda: n' ここで n はGPU 番号．Default to 'cpu'.
        """
        self.network = network
        # ---- amp setting ---- #
        self.use_amp = use_amp
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        # ---- use device setting ---- #
        if device_name == "cpu":
            self.device_name = device_name
            self.num_devices = []
        elif device_name == "cuda":
            self.device_name = device_name
            self.num_devices = [0]
        elif device_name == "auto":
            self.device_name, self.num_devices = SelectDevice()
        self.device = torch.device(self.device_name)
        # Dataparallel の使い方は以下のサイトを参照．
        # REF: https://qiita.com/m__k/items/87b3b1da15f35321ecf5
        if self.device_name == "cpu":
            self.network = DataParallel(self.network)
        else:
            self.network = DataParallel(self.network, device_ids=self.num_devices)
        # ---- multiple minibatch ---- #
        self.batch_multiplier = 3

    def reduce_loss_stats(self, loss_stats: dict) -> dict:
        """
        損失の統計情報を平均化する関数
        """
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch: int, data_loader, optimizer, recorder):
        self.network.train()
        max_iter = len(data_loader)
        multiple_mumibach_count = self.batch_multiplier  # multiple minibatch

        with tqdm(total=len(data_loader), leave=True, desc="train") as pbar:
            for iteration, batch in enumerate(data_loader):
                t_iter_start = time.time()
                iteration += 1
                recorder.step += 1

                # --------------- #
                # training stage #
                # --------------- #
                # optimizer の初期化
                optimizer.zero_grad()
                # use_amp = True の場合，混合精度を用いて訓練する．
                # 演算を混合精度でキャスト
                with amp.autocast(enabled=self.use_amp):
                    # if self.use_amp:  # もし，混合精度を使用する場合．
                    input = batch["img"].cuda(self.device)
                    target = batch["target"].cuda(self.device)
                    # non_blocking については以下を参照
                    # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
                    # input = batch["img"].half().cuda(self.device)
                    # target = batch["target"].long().cuda(self.device)
                    # else:  # 混合精度を使用しない場合
                    # input = batch["img"].float().cuda(self.device)
                    # target = batch["target"].long().cuda(self.device)

                    _, loss, loss_stats, image_stats = self.network(input, target)

                    loss = loss / self.batch_multiplier
                    print(loss)
                    """
                    if loss.ndim != 0:  # 損失の平均値を計算
                        loss = loss.mean()
                    else:  # バッチサイズをもとに平均値を計算
                        loss = loss / len(batch["cls_names"])
                    """

                # 損失をスケーリングし、backward()を呼び出してスケーリングされた微分を作成する
                self.scaler.scale(loss).backward()
                # グラデーションのスケールを解除し、optimizer.step()を呼び出すかスキップする。
                if multiple_mumibach_count == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    multiple_mumibach_count = self.batch_multiplier

                multiple_mumibach_count -= 1  # multiple minibatch

                batch_time = time.time() - t_iter_start
                recorder.batch_time.update(batch_time)
                recorder.update_loss_stats(loss_stats)
                pbar.update()

                if iteration % 3 == 0 or iteration == (max_iter - 1):
                    # print training state
                    # eta_seconds = recorder.batch_time.global_avg * (
                    #    max_iter - iteration
                    # )
                    # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
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
                    print(training_state)

                    # record loss_stats and image_dict
                    recorder.update_image_stats(image_stats)

                    recorder.record("train")

                # 【PyTorch】不要になった計算グラフを削除してメモリを節約
                # REF: https://tma15.github.io/blog/2020/08/22/pytorch%E4%B8%8D%E8%A6%81%E3%81%AB%E3%81%AA%E3%81%A3%E3%81%9F%E8%A8%88%E7%AE%97%E3%82%B0%E3%83%A9%E3%83%95%E3%82%92%E5%89%8A%E9%99%A4%E3%81%97%E3%81%A6%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84/
                # 誤差逆伝播を実行後、計算グラフを削除
                del input, target, loss, loss_stats, image_stats
                gc.collect()
                torch.cuda.empty_cache()

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        torch.cuda.empty_cache()
        self.network.eval()
        val_loss_stats = {}
        data_size = len(data_loader)
        with tqdm(total=len(data_loader), leave=True, desc="val") as pbar:
            for iteration, batch in enumerate(data_loader):
                with torch.no_grad():
                    with amp.autocast(enabled=self.use_amp):
                        # non_blocking については以下を参照
                        # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
                        # 【PyTorch】地味に知っておくべき実装の躓きドコロ
                        # REF: https://www.hellocybernetics.tech/entry/2018/02/20/182906
                        input = batch["img"].cuda(self.device)
                        target = batch["target"].cuda(self.device)

                        output, _, loss_stats, image_stats = self.network(input, target)

                    if evaluator is not None:
                        result = evaluator.evaluate(
                            iteration=iteration, batch_output=output, batch=batch
                        )
                    pbar.update()

            loss_stats = self.reduce_loss_stats(loss_stats)
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

        del input, target, output, loss_stats, image_stats
        gc.collect()
        torch.cuda.empty_cache()
