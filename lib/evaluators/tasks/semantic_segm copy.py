import gc
import os
import sys
from typing import Dict

sys.path.append("../../")
sys.path.append("../../../")


import numpy as np
import torch

from lib.utils.confusion import ConfusionMetrics
from lib.visualizers.segmentation import visualize


class SegmentationEvaluator(object):
    def __init__(self, result_dir: str) -> None:
        super(SegmentationEvaluator, self).__init__()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.outputs = []
        self.targets = []
        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1 = []

    def evaluate(self, iteration: int, batch_output: torch.Tensor, batch: Dict):
        for i in range(batch["img"].size()[0]):
            # バッチから 1 枚の画像を分離
            image = batch["img"][i, :, :, :]
            target = batch["target"][i, :, :]
            output = batch_output[i, :, :, :]

            fig = visualize(input=image, ground_truth_mask=target, predict=output)

            # fig.show()
            f_name = os.path.join(self.result_dir, "{}_{}.png")
            fig.savefig(f_name.format(str(iteration), str(i)), format="png")

            del fig
            gc.collect()
        # ---------------- #
        # 評価指標を計算 #
        # ---------------- #

        return None

    def summarize(self):
        acc = np.mean(self.accuracy)
        rec = np.mean(self.recall)
        pre = np.mean(self.precision)
        f1 = np.mean(self.f1)

        ConfusionMetrics(self.result_dir, self.targets, self.outputs)

        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1 = []

        return {"accuracy": acc, "recall": rec, "precision": pre, "f1": f1}
