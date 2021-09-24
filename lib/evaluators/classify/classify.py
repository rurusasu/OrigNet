import os
import sys

sys.path.append("../../")
sys.path.append("../../../")

from sklearn.metrics import accuracy_score


class evaluator(object):
    def __init__(self, result_dir: str) -> None:
        super(evaluator, self).__init__()
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.acc = []

    def evaluate(self, output, batch):
        acc = accuracy_score(batch["target"], output)
        self.acc.append(acc)
        return None
