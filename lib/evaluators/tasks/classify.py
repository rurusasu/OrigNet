import os
import sys

sys.path.append("../../")
sys.path.append("../../../")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class evaluator(object):
    def __init__(self, result_dir: str) -> None:
        super(evaluator, self).__init__()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1 = []

    def evaluate(self, output, batch):
        output = output.cpu().data.numpy()
        output = np.argmax(output, axis=1)
        target = batch["target"].cpu().data.numpy()

        # ---------------- #
        # 評価指標を計算 #
        # ---------------- #
        # 正解率
        acc_score = accuracy_score(target, output)
        self.acc.append(acc_score)
        # 再現率
        rec_score = recall_score(target, output, average=None)
        self.recall.append(rec_score)
        # 適合率
        pre_score = precision_score(target, output, average=None)
        self.precision.append(pre_score)
        # F値
        f1 = f1_score(target, output, average=None)
        self.f1.append(f1)

        # ----------------------- #
        # 混同行列の画像を作成 #
        # ----------------------- #
        cm = confusion_matrix(y_test, pred)

        sns.heatmap(cm, square=True, cbar=True, annot=True, cmap="Blues")
        plt.savefig("sklearn_confusion_matrix.png")

        return None
