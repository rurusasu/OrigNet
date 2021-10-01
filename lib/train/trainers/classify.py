from typing import Dict

import torch
import torch.nn as nn
from yacs.config import CfgNode


class NetworkWrapper(nn.Module):
    """
    ネットワークからの出力と教師データに基づいて損失を計算する機能をラップする
    画像を入力，出力をそのクラスラベルとする画像分類に特化したモデルを作成する
    """

    def __init__(self, cfg: CfgNode, net):
        super(NetworkWrapper, self).__init__()

        if "train" not in cfg and "criterion" not in cfg.train:
            raise ("The required parameter for `NetworkWrapper` is not set.")
        self.net = net
        self.criterion = cfg.train.criterion
        self.num_classes = cfg.num_classes

        # lenarning_typeによって
        if "mse" in self.criterion:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: Dict):
        input = batch["img"].cuda()
        target = batch["target"].cuda()

        # 出力は，
        # [[0番目のクラス，0番目のクラス，...batchの大きさ分繰り返し],
        #  [1番目のクラス，1番目のクラス， ...batchの大きさ分繰り返し]]
        output = self.net(input)

        # スカラステータス（）
        scalar_stats = {}
        loss = 0

        loss = self.criterion(output, target)

        axis = 1
        _, preds = torch.max(output, axis)
        acc = torch.sum(preds == target) / len(batch["cls_names"])

        scalar_stats.update({"batch_loss": loss, "batch_acc": acc})
        return output, loss, scalar_stats
