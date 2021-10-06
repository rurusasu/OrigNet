import sys

sys.path.append("../../")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.evaluators.tasks.classify import ClassifyEvaluator
from lib.evaluators.tasks.semantic_segm import SegmentationEvaluator


_evaluator_factory = {
    "classify": ClassifyEvaluator,
    "semantic_segm": SegmentationEvaluator,
}


def make_evaluator(cfg: CfgNode) -> object:
    """ネットワークの精度の検証を行うクラスを読みだす関数

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．

    Returns:
        object: ネットワークの精度の検証を行うクラス
    """
    if cfg.skip_eval:
        return None
    else:
        task = cfg.task
        eval_class = _evaluator_factory[task]
        eval_class = eval_class(cfg.result_dir)
        return eval_class
