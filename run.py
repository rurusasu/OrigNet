import argparse
import os
import sys
import yaml
from typing import Any, List

sys.path.append(".")
sys.path.append("..")

from yacs.config import CfgNode

from train import train
from test import test
from lib.config.config import pth, cfg
from lib.utils.base_utils import DirCheckAndMake


# parser = argparse.ArgumentParser()
# parser.add_argument("--up_file", type=str, default="", help="更新するパラメタのYAMLファイルパス")
# args = parser.parse_args()


def OneTrain(cfg: CfgNode, root_dir: str):
    """
    1回，train.py と test.py を実行する関数．
    訓練情報を保存するディレクトリは，
    cfg.task
        |
        |---- model
        |---- record
        |---- result

    Args:
        cfg (CfgNode): 訓練の条件設定が保存された辞書．
        root_dir (str): 親ディレクトリのパス．
    """
    train_dir = _OneTrainDir(cfg, root_dir)
    [mdl_dir, rec_dir, res_dir] = _OneTrainLogDir(cfg, train_dir)

    # コンフィグの更新
    cfg.model_dir = mdl_dir
    cfg.record_dir = rec_dir
    cfg.result_dir = res_dir

    # 訓練
    train(cfg)
    # テスト
    test(cfg)
    # Cfg 情報の保存
    _CfgSave(cfg, train_dir)


def _CfgSave(cfg: CfgNode, save_dir: str) -> Any:
    """CfgNode を yaml ファイルとして保存するための関数．

    Args:
        cfg (CfgNode): 訓練の条件設定が保存された辞書．
        save_dir (str): CfgNode を保存するための yaml ファイルのパス．
    """
    dic = {}
    w_pth = os.path.join(save_dir, f"{cfg.task}_{cfg.model}.yaml")
    # CfgNode -> Dict
    # この変換をしない場合，不要な変数が YAML に保存される．
    for k, v in cfg.items():
        dic[k] = v

    with open(w_pth, "w") as yf:
        yaml.dump(dic, yf, default_flow_style=False)


def _OneTrainLogDir(cfg: CfgNode, root_dir: str = ".") -> List[str]:
    """
    1回の訓練と検証時のデータを保存する`model`，`record`，`result` ディレクトリを作成するための関数．

    Args:
        cfg (CfgNode): 訓練の条件設定が保存された辞書．
        root_dir (str): 親ディレクトリのパス．
    """
    # もしモデル保存用ディレクトリが設定されていなかった場合．
    if "model_dir" not in cfg:
        cfg.model_dir = "model"
    if "record_dir" not in cfg:
        cfg.record_dir = "record"
    if "result_dir" not in cfg:
        cfg.result_dir = "result"

    model_dir = DirCheckAndMake(os.path.join(root_dir, cfg.model_dir))
    record_dir = DirCheckAndMake(os.path.join(root_dir, cfg.record_dir))
    result_dir = DirCheckAndMake(os.path.join(root_dir, cfg.result_dir))

    return [model_dir, record_dir, result_dir]


def _OneTrainDir(cfg: CfgNode, root_dir: str = ".") -> str:
    """
    1回の訓練の全データを保存するディレクトリを作成する関数．
    ディレクトリ名は，cfg.task で与えられる．

    Args:
        root_dir (str): 親ディレクトリのパス．
    """
    if "task" not in cfg:
        raise ("The task is not set.")
    dir_pth = os.path.join(root_dir, cfg.task)
    dir_pth = DirCheckAndMake(dir_pth)
    return dir_pth


def _make_learning_dir(dir_name: str = "trained") -> str:
    """
    訓練と検証の全データを保存するディレクトリを作成する関数

    Arg:
        dir_name (str):  訓練と検証の全データを保存するディレクトリ名

    Return:
        dir_pth (str):  訓練と検証の全データを保存するディレクトリのパス．
    """
    dir_pth = os.path.join(pth.DATA_DIR, dir_name)
    dir_pth = DirCheckAndMake(dir_pth)
    return dir_pth


class CycleTrain(object):
    def __init__(self, cfg) -> None:
        super(CycleTrain, self).__init__()
        # self.args = args
        self.up_file = os.path.join(pth.CONFIGS_DIR, "update.yaml")
        self.cfg = cfg
        self.root_dir = _make_learning_dir()

    def UpdataCfg(self, iter_num: int = 1) -> bool:
        """
        訓練のコンフィグをアップデートする関数．

        Args:
            iter_num (int, optional): 訓練サイクルの現在のイテレーション数. Defaults to 1.
        """
        # 追加の訓練サイクルが存在するか判定する変数
        # サイクルが最後まで行った or up_file not found: False
        # それ以外の場合: True
        hook = False
        # アップデート前にコンフィグ情報をコピー

        yml_pth = self.up_file
        with open(yml_pth, "r") as yml:
            # YAML ファイルからコンフィグ情報を読みだす．
            # REF: https://qiita.com/seigot/items/10329d1cdf8b12cf7749
            new_cfg = yaml.safe_load(yml)
            for dic in new_cfg:
                # 訓練サイクルのイテレーション数と
                # アップデート情報のイテレーション数が同じ場合
                if dic["loop"] == iter_num:
                    for k, v in dic.items():
                        # イテレーション数の情報は訓練に不要なので飛ばす．
                        if k == "loop":
                            pass
                        # 訓練情報をアップデートする．
                        else:
                            # v が辞書型の場合
                            # if type(v) == "dict":
                            if isinstance(v, dict):
                                for v_key, v_value in v.items():
                                    self.cfg[k][v_key] = v_value
                            else:
                                self.cfg.merge_from_list([k, v])
                        hook = True
                else:
                    pass

        return hook

    def main(self):
        # update用のコンフィグが指定されなかった場合．
        if not self.up_file:
            print(f"{self.args.cfg_file} に設定された情報を用いて訓練・検証を実行します．")
            OneTrain(self.cfg, root_dir=self.root_dir)

        # update用のコンフィグ情報が指定された場合．
        else:
            print(f"{self.up_file} を探しています．")
            # ファイルが実際に存在するか判定
            if os.path.exists(self.up_file):
                print("ファイルが見つかりました．")
                print(f"{self.up_file} の情報を用いて連続訓練を開始します．")

                iter_num = 1
                # 追加の訓練サイクルが存在するか判定する変数
                # サイクルが最後まで行った or up_file not found: False
                # それ以外の場合: True
                hock = False
                while True:
                    print(f"{iter_num} 番目の学習を実行します。")
                    dir = DirCheckAndMake(os.path.join(self.root_dir, str(iter_num)))
                    OneTrain(self.cfg, root_dir=dir)
                    hock = self.UpdataCfg(iter_num)
                    if hock:
                        iter_num += 1
                    else:
                        break

                print("連続訓練が終了しました．")
            else:
                print("ファイルが見つかりました．")
                print("連続訓練を停止します．")


if __name__ == "__main__":
    debug = False
    if not debug:
        CycleTrain(cfg).main()
    else:
        from yacs.config import CfgNode as CN

        conf = CN()
        conf.task = "classify"
        conf.network = "cnns"
        conf.model = "inc_v3"
        conf.model_dir = "model"
        conf.train_type = "transfer"  # or scratch
        # conf.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.use_amp = False  # 半精度で訓練するか
        conf.record_dir = "record"
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
        conf.skip_eval = False
        conf.train = CN()
        conf.train.epoch = 1
        conf.train.dataset = "SampleTrain"
        # conf.train.dataset = "AngleDetectTrain_2"
        conf.train.batch_size = 20
        conf.train.num_workers = 2
        conf.train.batch_sampler = ""
        conf.train.optim = "adam"
        conf.train.criterion = ""
        conf.train.lr = 1e-3
        conf.train.scheduler = "step_lr"
        conf.train.weight_decay = 0.0
        conf.train.milestones = (20, 40, 60, 80, 100, 120, 160, 180, 200, 220)
        conf.train.warp_iter = 10
        conf.train.gamma = 0.5
        conf.val = CN()
        conf.val.dataset = "SampleTest"
        conf.val.batch_size = 20
        conf.val.num_workers = 2
        conf.val.batch_sampler = ""
        conf.test = CN()
        # conf.test.dataset = "AngleDetectVal_2"
        conf.test.dataset = "SampleTest"
        conf.test.batch_size = 20
        conf.test.num_workers = 2
        conf.test.batch_sampler = ""

        CycleTrain(conf).main()
