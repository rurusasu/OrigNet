import os
import sys

sys.path.append("../../")
sys.path.append("../../../")


import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

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

    def evaluate(self, output, batch):
        # output = output.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        # target = batch["target"].cpu().data.numpy()
        for i in range(batch["img"].size()[0]):
            # バッチから 1 枚の画像を分離
            image = batch["img"][i, :, :, :]
            target = batch["target"][i, :, :, :]
            output = output[i, :, :, :]

            visualize(image=image, ground_truth_mask=target, predict=output)

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
