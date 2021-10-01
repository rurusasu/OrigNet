import datetime
from re import S
import sys
import time

sys.path.append("../")

import torch
import tqdm
from torch.nn import DataParallel
from torch.cuda import amp

from lib.utils.base_utils import SelectDevice


class Trainer(object):
    def __init__(self, network, device: str = "cpu"):
        """
        device 引数について不明な場合は以下を参照．
        REF: https://note.nkmk.me/python-pytorch-device-to-cuda-cpu/

        Args:
            network: 訓練されるネットワーク
            device(str): 'cpu' もしくは 'cuda: n' ここで n はGPU 番号．Default to 'cpu'.
        """

        num_device = SelectDevice()
        if num_device == "cpu":
            network = DataParallel(network)
        else:
            network = DataParallel(network, device_ids=num_device)
        self.network = network.to(device)
        self.scaler = amp.GradScaler()
        self.device = torch.device(device)

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
                """
                # 混合精度テスト
                # optimizer の初期化
                optimizer.zero_grad()
                # 演算を混合精度でキャスト
                with amp.autocast():
                    output, loss, loss_stats = self.network(batch)
                    if loss.ndim != 0:
                        # 損失の平均値を計算
                        loss = loss.mean()
                    else:
                        # バッチサイズをもとに平均値を計算
                        loss = loss / len(batch["cls_names"])

                # 損失をスケーリングし、backward()を呼び出してスケーリングされた微分を作成する
                self.scaler.scale(loss).backward()

                # グラデーションのスケールを解除し、optimizer.step()を呼び出すかスキップする。
                self.scaler.step(optimizer)
                optimizer.step()
                self.scaler.update()
                """

                output, loss, loss_stats = self.network(batch)
                if loss.ndim != 0:
                    # 損失の平均値を計算
                    loss = loss.mean()
                else:
                    # バッチサイズをもとに平均値を計算
                    loss = loss / len(batch["cls_names"])
                # optimizer の初期化
                optimizer.zero_grad()
                # 逆伝搬計算
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
                # パラメタ更新
                optimizer.step()

                # data recording stage
                loss_stats = self.reduce_loss_stats(loss_stats)
                recorder.update_loss_stats(loss_stats)

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
            # batch["img"] = batch["img"].cuda()
            # if batch["msk"]:
            #     batch["msk"] = batch["msk"].cuda()

            with torch.no_grad():
                output, loss, loss_stats = self.network(batch)
                if evaluator is not None:
                    result = evaluator.evaluate(output=output, batch=batch)

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
