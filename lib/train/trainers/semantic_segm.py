import torch
import torch.nn as nn
from yacs.config import CfgNode

from lib.train.metricses import make_metrics


class NetworkWrapper(nn.Module):
    """
    ネットワークからの出力と教師データに基づいて損失を計算する機能をラップする
    画像を入力，出力をそのクラスラベルとする画像分類に特化したモデルを作成する
    """

    def __init__(self, cfg: CfgNode, network):
        super(NetworkWrapper, self).__init__()

        if "train" not in cfg and "criterion" not in cfg.train:
            raise ("The required parameter for `NetworkWrapper` is not set.")
        self.network = network

        # 損失関数 (criterion) を選択
        if "mse" in cfg.train.criterion:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 評価指標 (metrics) を選択
        self.metrics = make_metrics(cfg)

    def forward(self, batch: int):
        output = self.network(batch["img"])
        # スカラステータス（）
        scalar_stats = {}
        loss = 0

        if "test" in batch["meta"]:
            loss = torch.tensor(0).to(batch["img"].device)
            return output, loss, {}

        loss = self.criterion(output, batch["cls_num"])
        iou = self.metrics(output[-1], batch["msk"])

        scalar_stats.update({"loss": loss})
        return output, loss, scalar_stats
