import datetime
import sys
import time
from typing import Literal

sys.path.append("../")

import torch
import tqdm
from torch.autograd import Variable
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
        if device_name == "cpu":
            self.device_name = device_name
            self.num_devices = []
        elif device_name == "cuda":
            self.device_name = device_name
            self.num_devices = [0]
        elif device_name == "auto":
            self.device_name, self.num_devices = SelectDevice()
        # Dataparallel の使い方は以下のサイトを参照．
        # REF: https://qiita.com/m__k/items/87b3b1da15f35321ecf5
        if self.device_name == "cpu":
            network = DataParallel(network)
        else:
            network = DataParallel(network, device_ids=self.num_devices)
        # else: # 複数の GPU が搭載されている場合．未実装
        # network = DataParallel(network, device=self.num_device)
        self.device = torch.device(self.device_name)
        self.network = network.to(device=self.device)
        # ---- amp setting ---- #
        self.use_amp = use_amp
        self.scaler = amp.GradScaler(enabled=self.use_amp)

    def reduce_loss_stats(self, loss_stats: dict) -> dict:
        """
        損失の統計情報を平均化する関数
        """
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def train(self, epoch: int, data_loader, optimizer, recorder):
        torch.cuda.empty_cache()
        self.network.train()
        max_iter = len(data_loader)
        end = time.time()

        with tqdm.tqdm(total=len(data_loader), leave=False, desc="train") as pbar:
            for iteration, batch in enumerate(data_loader):
                data_time = time.time() - end
                iteration += 1
                recorder.step += 1

                # --------------- #
                # training stage #
                # --------------- #
                # 混合精度テスト
                # optimizer の初期化
                optimizer.zero_grad()
                # 演算を混合精度でキャスト
                with amp.autocast(enabled=self.use_amp):
                    # もし，混合精度を使用する場合．
                    if self.use_amp:
                        # non_blocking については以下を参照
                        # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
                        input = batch["img"].to(
                            device=self.device, dtype=torch.float16, non_blocking=True
                        )
                        target = batch["target"].to(
                            device=self.device, dtype=torch.float16, non_blocking=True
                        )
                    # 混合精度を使用しない場合
                    else:
                        input = batch["img"].to(device=self.device, non_blocking=True)
                        target = batch["target"].to(
                            device=self.device, non_blocking=True
                        )

                    output, loss, loss_stats = self.network(input, target)

                    if loss.ndim != 0:
                        # 損失の平均値を計算
                        loss = loss.mean()
                    else:
                        # バッチサイズをもとに平均値を計算
                        loss = loss / len(batch["cls_names"])

                # 損失をスケーリングし、backward()を呼び出してスケーリングされた微分を作成する
                self.scaler.scale(loss).backward()

                # 【PyTorch】不要になった計算グラフを削除してメモリを節約
                # REF: https://tma15.github.io/blog/2020/08/22/pytorch%E4%B8%8D%E8%A6%81%E3%81%AB%E3%81%AA%E3%81%A3%E3%81%9F%E8%A8%88%E7%AE%97%E3%82%B0%E3%83%A9%E3%83%95%E3%82%92%E5%89%8A%E9%99%A4%E3%81%97%E3%81%A6%E3%83%A1%E3%83%A2%E3%83%AA%E3%82%92%E7%AF%80%E7%B4%84/
                del input, output, target, loss, batch  # 誤差逆伝播を実行後、計算グラフを削除

                # グラデーションのスケールを解除し、optimizer.step()を呼び出すかスキップする。
                self.scaler.step(optimizer)
                # optimizer.step()
                self.scaler.update()

                # data recording stage
                # loss_stats = self.reduce_loss_stats(loss_stats)
                recorder.update_loss_stats(loss_stats)
                del loss_stats

                batch_time = time.time() - end
                end = time.time()
                recorder.batch_time.update(batch_time)
                recorder.data_time.update(data_time)
                pbar.update()

                if iteration % 20 == 0 or iteration == (max_iter - 1):
                    # print training state
                    eta_seconds = recorder.batch_time.global_avg * (
                        max_iter - iteration
                    )
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    lr = optimizer.param_groups[0]["lr"]
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                    training_state = "  ".join(
                        ["eta: {}", "{}", "lr: {:.6f}", "max_mem: {:.0f}"]
                    )
                    training_state = training_state.format(
                        eta_string, str(recorder), lr, memory
                    )
                    print(training_state)

                    # recod loss_stats and img_dict
                    # recorder.update_image_stats(image_stats)
                    recorder.record("train")

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        torch.cuda.empty_cache()
        self.network.eval()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp):
                    if self.use_amp:
                        # non_blocking については以下を参照
                        # REF: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
                        # 【PyTorch】地味に知っておくべき実装の躓きドコロ
                        # REF: https://www.hellocybernetics.tech/entry/2018/02/20/182906
                        input = batch["img"].to(
                            device=self.device, dtype=torch.float16, non_blocking=True
                        )
                        target = batch["target"].to(
                            device=self.device, dtype=torch.float16, non_blocking=True
                        )
                    # 混合精度を使用しない場合
                    else:
                        input = batch["img"].to(device=self.device, non_blocking=True)
                        target = batch["target"].to(
                            device=self.device, non_blocking=True
                        )

                    output, _, loss_stats = self.network(input, target)
                if evaluator is not None:
                    result = evaluator.evaluate(
                        input=input, output=output, target=target
                    )

            del input, output, target, batch

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
            recorder.record("val", epoch, val_loss_stats)
        del val_loss_stats
