import os
import sys

sys.path.append("../../")
sys.path.append("../../../")

from sklearn.metrics import accuracy_score
from yacs.config import CfgNode


class Evaluator:
    def __init__(self, cfg: CfgNode, network, result_dir: str) -> None:
        self.result_dir = os.path.join(result_dir, cfg.test.dataset)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.network = network

        self.acc = []

    def evaluate(self, output, batch):
        acc = accuracy_score(batch["target"], output)
        self.acc.append(acc)
