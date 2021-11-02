import sys
from typing import Literal


sys.path.append("../../../")

import torch
from torch.cuda import amp
from torch.nn import DataParallel

from lib.utils.base_utils import SelectDevice
from lib.utils.net_utils import train, val


class Trainer(object):
    def __init__(
        self,
        network,
        device_name: Literal["cpu", "cuda", "auto"] = "auto",
        use_amp: bool = True,
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

    def train(self, epoch: int, data_loader, optimizer, recorder):
        train(
            self.network,
            epoch=epoch,
            data_loader=data_loader,
            optimizer=optimizer,
            recorder=recorder,
            device=self.device,
            use_amp=self.use_amp,
            batch_multiplier=self.batch_multiplier,
        )

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        val_loss = val(
            self.network,
            epoch=epoch,
            data_loader=data_loader,
            evaluator=evaluator,
            recorder=recorder,
            device=self.device,
            use_amp=self.use_amp,
        )

        return val_loss
