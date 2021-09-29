import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def ConfusionMetrics(result_dir: str, target: np.ndarray, output: np.ndarray) -> None:
    # ----------------------- #
    # 混同行列の画像を作成 #
    # ----------------------- #
    cm = confusion_matrix(target, output)

    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap="Blues")
    plt.xlabel("Pre", fontsize=13)
    plt.ylabel("GT", fontsize=13)
    plt.savefig(os.path.join(result_dir, "sklearn_confusion_matrix.png"))

    return None
