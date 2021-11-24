import sys

sys.path.append("../../")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.utils.metrics.CrossEntropyLoss import CrossEntropyLoss
from lib.utils.metrics.dice import DiceLoss
from lib.utils.metrics.iou import iou_score


_metrics_factory = {"iou": iou_score, "dice": DiceLoss, "cel": CrossEntropyLoss}


def make_metrics(cfg: CfgNode) -> object:
    """semantic segmentation で使用する評価指標(metrics)の関数を返す関数．

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．

    Returns:
        : `utils.metrics` 内に保存された metrics class の内の1つ．
    """
    if "train" not in cfg and "criterion" not in cfg.train:
        raise ("The required parameter `criterion` for `make_metrics` is not set.")

    metrics = _metrics_factory[cfg.train.criterion]()
    # metrics = metrics()
    return metrics


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.train = CfgNode()
    cfg.train.metrics = "cel"

    f = make_metrics(cfg)
