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

        # lenarning_typeによって
        if "mse" in self.criterion:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: int):
        output = self.net(batch["img"])
        # スカラステータス（）
        scalar_stats = {}
        loss = 0

        if "test" in batch["meta"]:
            loss = torch.tensor(0).to(batch["img"].device)
            return output, loss, {}

        loss = self.criterion(output, batch["cls_num"])

        scalar_stats.update({"loss": loss})
        return output, loss, scalar_stats
