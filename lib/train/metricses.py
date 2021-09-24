from typing import Any

from yacs.config import CfgNode

from lib.utils.metrics.iou import iou_score, dice_coef


_metrics_factory = {"iou": iou_score, "dice": dice_coef}


def make_metrics(cfg: CfgNode) -> Any:
    """semantic segmentation で使用する評価指標(metrics)の関数を返す関数．

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．

    Returns:
        Any: `utils.metrics` 内に保存された metrics class の内の1つ．
    """
    if "train" not in cfg and "metrics" not in cfg.train:
        raise ("The required parameter for `make_metrics` is not set.")

    metrics = _metrics_factory[cfg.train.metrics]
    metrics = metrics(**cfg)
    return metrics
