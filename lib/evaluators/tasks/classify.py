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


class ClassifyEvaluator(object):
    def __init__(self, result_dir: str) -> None:
        super(ClassifyEvaluator, self).__init__()
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
        output = output.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        target = batch["target"].cpu().data.numpy()

        # ---------------- #
        # 評価指標を計算 #
        # ---------------- #
        # `batch`ごとの分類結果を表示
        report = classification_report(target, output, target_names=batch["cls_names"])
        print(report)
        # 正解率
        acc_score = accuracy_score(target, output)
        self.accuracy.append(acc_score)
        # 再現率
        rec_score = recall_score(target, output, average=None)
        self.recall.append(rec_score)
        # 適合率
        pre_score = precision_score(target, output, average=None)
        self.precision.append(pre_score)
        # F値
        f1 = f1_score(target, output, average=None)
        self.f1.append(f1)

        # self.outputs.append(output[:])
        self.outputs = np.hstack((self.outputs, output))
        self.targets = np.hstack((self.targets, target))

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
