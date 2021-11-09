import os
import random
import sys
import yaml

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
from yacs.config import CfgNode

from train import train
from test import test
from lib.config.config import pth, cfg
from lib.train.trainers.OptunaTrainer import OptunaTrainer
from lib.utils.base_utils import CfgSave, DirCheckAndMake, OneTrainDir, OneTrainLogDir

seed = 2021


def fix_seed(seed: int):
    """
    乱数の seed を固定するための関数．

    Args:
        seed (int): 乱数を固定するための整数値．
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def OneTrain(config: CfgNode, root_dir: str):
    """
    1回，train.py と test.py を実行する関数．
    訓練情報を保存するディレクトリは，
    cfg.task
        |
        |---- model
        |---- record
        |---- result

    Args:
        config (CfgNode): 訓練の条件設定が保存された辞書．
        root_dir (str): 親ディレクトリのパス．
    """
    train_dir = OneTrainDir(root_dir, dir_name=config.task)
    [mdl_dir, rec_dir, res_dir] = OneTrainLogDir(config, train_dir)

    # コンフィグの更新
    config.model_dir = mdl_dir
    config.record_dir = rec_dir
    config.result_dir = res_dir

    # 訓練
    train(config)
    # テスト
    loss = test(config)
    # Cfg 情報の保存
    CfgSave(config, train_dir)

    return loss


def make_learning_dir(dir_name: str = "trained") -> str:
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
    def __init__(self, config) -> None:
        super(CycleTrain, self).__init__()
        self.config = config
        if "optuna" not in self.config:
            raise ("Required parameters `optuna` for CycleTrain are not set.")
        self.up_file = os.path.join(pth.CONFIGS_DIR, "update.yaml")

        # 乱数の seed を固定
        fix_seed(seed)

        # ------------------------------------
        # ディレクトリ設定
        # ------------------------------------
        self.root_dir = make_learning_dir()
        # もしモデル保存用ディレクトリが設定されていなかった場合．
        if "model_dir" not in self.config:
            self.config.model_dir = "model"
        if "record_dir" not in self.config:
            self.config.record_dir = "record"
        if "result_dir" not in self.config:
            self.config.result_dir = "result"
        self.model_dir = self.config.model_dir
        self.record_dir = self.config.record_dir
        self.result_dir = self.config.result_dir
        # ------------------------------------

        if self.config.optuna:
            if "optuna_trials" not in self.config or self.config.optuna_trials < 0:
                raise (
                    "Required parameters `optuna_trials` for CycleTrain are not set."
                )
            # パラメタ探索のため，追加学習を無効にする．
            self.config.resume = False
            self.opt_train = OptunaTrainer(self.config)

        self.iter_num = 1

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
                                    self.config[k][v_key] = v_value
                            else:
                                self.config.merge_from_list([k, v])
                        hook = True
                else:
                    pass

        return hook

    def main(self):
        # update用のコンフィグが指定されなかった場合．
        if not self.up_file:
            print(f"{self.args.cfg_file} に設定された情報を用いて訓練・検証を実行します．")
            if self.config.optuna:
                self.opt_train.Train(root_dir=self.root_dir)
            else:
                OneTrain(self.config, root_dir=self.root_dir)

        # update用のコンフィグ情報が指定された場合．
        else:
            print(f"{self.up_file} を探しています．")
            # ファイルが実際に存在するか判定
            if os.path.exists(self.up_file):
                print("ファイルが見つかりました．")
                print(f"{self.up_file} の情報を用いて連続訓練を開始します．")

                self.iter_num = 1
                # 追加の訓練サイクルが存在するか判定する変数
                # サイクルが最後まで行った or up_file not found: False
                # それ以外の場合: True
                hock = False

                # --------------------------
                # 連続訓練のメイン部分
                # --------------------------
                while True:
                    print(f"{self.config.model} の学習を実行します。")
                    # イテレーションごとのルートディレクトリの作成
                    dir = DirCheckAndMake(
                        os.path.join(self.root_dir, str(self.iter_num))
                    )

                    # ---------------------#
                    # Training & Testing #
                    # ---------------------#
                    if self.config.optuna:
                        self.opt_train.Train(root_dir=dir)
                    else:
                        OneTrain(self.config, root_dir=dir)

                    hock = self.UpdataCfg(self.iter_num)
                    if hock:
                        # イテレーション数の更新
                        self.iter_num += 1
                        # 保存用の子ディレクトリの再設定
                        # これをしないと，パスが深くなり続けてしまう。
                        self.config.model_dir = self.model_dir
                        self.config.record_dir = self.record_dir
                        self.config.result_dir = self.result_dir
                    else:
                        break

                print("連続訓練が終了しました．")
            else:
                print("update.yaml ファイルが見つかりませんでした．")
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
        # conf.model = "inc_v3"
        # conf.model = "vgg_11_bn"
        conf.model = "inc_res_v2"
        conf.model_dir = "model"
        conf.train_type = "transfer"  # or scratch
        conf.replaced_layer_num = 1  # 転移学習で置き換える全結合層の数
        # conf.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.use_amp = False  # 半精度で訓練するか
        conf.optuna = True
        conf.optuna_trials = 5
        conf.record_dir = "record"
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
        conf.skip_eval = False
        conf.train = CN()
        conf.train.epoch = 5
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
        # conf.val.dataset = "AngleDetectTrain_2"
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
