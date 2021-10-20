import os
import sys
from typing import Dict, List

from tensorboardX import writer

sys.path.append("../../")
sys.path.append("../../../")


import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
import torch

from lib.utils.confusion import ConfusionMetrics


class ClassifyEvaluator(object):
    def __init__(
        self,
        result_dir: str,
        cls_names: List[str],
    ) -> None:
        super(ClassifyEvaluator, self).__init__()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.cls_names = cls_names
        self.cls_labels = np.arange(len(self.cls_names))

        self.outputs = []
        self.targets = []
        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1 = []

        # ---- 結果を書き込むための Excel の設定 ---- #
        # book を作成
        # self.writer = pd.ExcelWriter(os.path.join(self.result_dir, "result.xlsx"))
        self.wb = openpyxl.Workbook()
        self.sheet = self.wb.active
        self.row = 1

    def evaluate(
        self,
        iteration: int,
        batch_output: torch.Tensor,
        batch: Dict,
    ):
        output = batch_output.detach().clone().cpu().numpy()
        # output = np.argmax(output, axis=1)
        target = batch["target"].detach().clone().cpu().data.numpy()

        # ---------------- #
        # 評価指標を計算 #
        # ---------------- #
        # `batch`ごとの分類結果を表示
        # report = classification_report(
        #     target.tolist(), output.tolist(), labels=self.cls_labels
        # )
        # print(report)

        # 正解率
        acc_score = accuracy_score(target, output)
        self.accuracy.append(acc_score)
        # 再現率
        rec_score = recall_score(target, output, average="micro")
        self.recall.append(rec_score)
        # 適合率
        pre_score = precision_score(target, output, average="micro")
        self.precision.append(pre_score)
        # F値
        f1 = f1_score(target, output, average="micro")
        self.f1.append(f1)

        data = {
            "acc": acc_score.tolist(),
            "rec": rec_score.tolist(),
            "pre": pre_score.tolist(),
            "f1": f1.tolist(),
        }

        for key in data.keys():
            # worksheet.write(row, 0, key)
            # worksheet.write_row(row, 1, myDictionary[key])
            self.sheet.cell(row=self.row, column=2).value = data[key]
            self.row += 1

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
        self.wb.save(os.path.join(self.result_dir, "result.xlsx"))

        return {"accuracy": acc, "recall": rec, "precision": pre, "f1": f1}
