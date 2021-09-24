import os

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
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)

    return my_module.Dataset
