import sys

sys.path.append("../../../")

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.autograd import Variable
from yacs.config import CfgNode

from lib.train.metricses import make_metrics


class SemanticSegmentationNetworkWrapper(nn.Module):
    """
    ネットワークからの出力と教師データに基づいて損失を計算する機能をラップする
    画像を入力，出力をそのクラスラベルとする画像分類に特化したモデルを作成する
    """

    def __init__(self, cfg: CfgNode, net):
        super(SemanticSegmentationNetworkWrapper, self).__init__()

        if "train" not in cfg and "criterion" not in cfg.train:
            raise (
                "The required parameter for `SemanticSegmentationNetworkWrapper` is not set."
            )

        self.net = net

        # 損失関数 (criterion) を選択
        self.criterion = smp.utils.losses.DiceLoss()

        # 評価指標 (metrics) を選択
        # self.metrics = make_metrics(cfg)
        self.metrics = smp.utils.metrics.IoU()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        output = self.net.forward(input)
        # スカラステータス（）
        loss = 0
        iou = 0
        image_stats = {}
        scalar_stats = {}

        loss = self.criterion(output, target)
        iou = self.metrics(output, target)

        image_stats.update(
            {
                "input": input.detach().clone(),
                "target": target.detach().clone(),
                "output": output.detach().clone(),
            }
        )

        scalar_stats.update(
            {"batch_loss": loss.detach().clone(), "batch_iou": iou.detach().clone()}
        )

        del input, target, iou  # loss と iou 計算後 batch を削除してメモリを確保
        torch.cuda.empty_cache()

        return output, loss, scalar_stats, image_stats


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.train = CfgNode()
    cfg.train.metrics = "iou"

    f = make_metrics(cfg)
    print(f)
