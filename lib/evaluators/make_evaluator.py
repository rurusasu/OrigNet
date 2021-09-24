import importlib.util
import os
import sys

sys.path.append("../../")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.config.config import pth


def _evaluator_factory(cfg: CfgNode) -> object:
    """データセット名に合わせて作成されたディレクトリ内のファイルを読みだす関数

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．

    Returns:
        object: 引数で指定した python ソースファイル内に記述されている関数
    """
    eval_pth = os.path.join(pth.LIB_DIR, "evaluators", "tasks", cfg.task + ".py")
    spec = importlib.util.spec_from_file_location(cfg.task, eval_pth)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)

    return eval_module.evaluator


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
        eval_class = _evaluator_factory(cfg)
        eval_class = eval_class(cfg.result_dir)
        return eval_class
