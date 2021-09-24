import importlib.util
import os
import sys

sys.path.append("../../")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.config.config import pth


def _evaluator_factory(pth, task: str) -> object:
    """データセット名に合わせて作成されたディレクトリ内のファイルを読みだす関数

    Args:
        data_source (str): DataCatalog に保存されているデータセット名と同じ文字列
        task (str): 実行するタスク名．例: 'classify'

    Returns:
        object: 引数で指定した python ソースファイル内に記述されている関数
    """
    pth = os.path.join(pth.LIB_DIR, "evaluators", "tasks", task + ".py")
    spec = importlib.util.spec_from_file_location(task, pth)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)

    return eval_module.Dataset


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
        return _evaluator_factory(cfg)
