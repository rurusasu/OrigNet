import torch
import torch.nn as nn
from yacs.config import CfgNode


class ClassifyNetworkWrapper(nn.Module):
    """
    ネットワークからの出力と教師データに基づいて損失を計算する機能をラップする
    画像を入力，出力をそのクラスラベルとする画像分類に特化したモデルを作成する
    """

    def __init__(self, cfg: CfgNode, net):
        super(ClassifyNetworkWrapper, self).__init__()

        if "train" not in cfg and "criterion" not in cfg.train:
            raise ("The required parameter for `NetworkWrapper` is not set.")
        self.net = net
        self.criterion = cfg.train.criterion

        # lenarning_typeによって
        if "mse" in self.criterion:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # 出力は，
        # [[0番目のクラス，0番目のクラス，...batchの大きさ分繰り返し],
        #  [1番目のクラス，1番目のクラス， ...batchの大きさ分繰り返し]]
        output = self.net(input)

        # スカラステータス（）
        loss = 0
        image_stats = {}
        scalar_stats = {}

        loss = self.criterion(output, target)

        # output の予測をラベル値に変換．
        # 例: [[1.09e+01, -6.46e+0.1, ...][1.96e+01, -8.70e+01, ...]] -> [0, 0, ...]
        _, preds = torch.max(output, axis=1)
        acc = torch.sum(preds == target) / input.size()[0]

        # image_stats.update({"input": input.detach().clone().cpu()})
        scalar_stats.update(
            {"batch_loss": loss.detach().clone(), "batch_acc": acc.detach().clone()}
        )

        del input, target, acc  # loss と iou 計算後 batch を削除してメモリを確保
        torch.cuda.empty_cache()

        return preds, loss, scalar_stats, image_stats
