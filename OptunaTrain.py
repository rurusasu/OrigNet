import gc
import os
import sys
import uuid

sys.path.append(".")
sys.path.append("../../")
sys.path.append("../../../")

import optuna
import torch
from yacs.config import CfgNode

from lib.config.config import pth, cfg
from lib.datasets.make_datasets import make_data_loader
from lib.models.make_network import make_network
from lib.train.scheduler import make_lr_scheduler
from lib.train.optimizers import make_optimizer
from lib.train.trainers.make_trainer import make_trainer
from lib.train.recorder import make_recorder
from lib.utils.base_utils import CfgSave, OneTrainDir, OneTrainLogDir
from lib.utils.net_utils import save_model
from lib.visualizers.NetVisualization import NetVisualization
from test import test


class OptunaTrain(object):
    def __init__(
        self,
        config: CfgNode,
    ) -> None:
        super(OptunaTrain, self).__init__()
        self.config = config
        if "train" not in self.config:
            raise ("Required parameters `train` for OptinaTrainer are not set.")
        if "optuna_trials" not in self.config or self.config.optuna_trials < 0:
            raise ("Required parameters `optuna_trials` for CycleTrain are not set.")
        self.max_trial_num = self.config.optuna_trials
        self.trial_count = 1

        # PyTorchが自動で、処理速度の観点でハードウェアに適したアルゴリズムを選択してくれます。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.multiprocessing.set_sharing_strategy("file_system")

        # 訓練と検証用のデータローダーを作成
        self.train_loader = make_data_loader(
            dataset_name=self.config.train.dataset,
            batch_size=self.config.train.batch_size,
            batch_sampler=self.config.train.batch_sampler,
            ds_category="train",
            img_shape={
                "width": self.config.img_width,
                "height": self.config.img_height,
            },
            max_iter=self.config.ep_iter,
            num_workers=self.config.train.num_workers,
            task=self.config.task,
        )
        self.val_loader = make_data_loader(
            dataset_name=self.config.val.dataset,
            batch_size=self.config.val.batch_size,
            batch_sampler=self.config.val.batch_sampler,
            ds_category="val",
            img_shape={
                "width": self.config.img_width,
                "height": self.config.img_height,
            },
            num_workers=self.config.val.num_workers,
            task=self.config.task,
        )

        # セマンティックセグメンテーションの場合，背景のクラスを追加しないと cross_entropy の計算でエラーが発生．
        if self.config.task == "classify":
            self.config.num_classes = len(self.train_loader.dataset.cls_names)
        elif self.config.task == "semantic_segm":
            # 理由は，画素値が 0 のラベルを与える必要があるため．
            self.config.num_classes = len(self.train_loader.dataset.cls_names) + 1
        else:
            raise ValueError(
                "Choose either `classify` or `semantic_segm` for the task."
            )
        self.mdl_dir = self.config.model_dir
        self.rec_dir = self.config.record_dir
        self.res_dir = self.config.result_dir

    def Train(self, root_dir: str = "."):
        """
        optuna を使用した最適なハイパーパラメタの探索を行う関数．\\
        REF: [optuna_doc](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls) \\
        REF: [sampler の seed 固定](https://qiita.com/c60evaporator/items/633575c37863d1d18335) \\
        REF: [storage への変数の渡し方](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html)

        Args:
            update_root_dir (str] optional): 最適化中の訓練情報を保存するディレクトリの親ディレクトリのパス. Defaults to ".".
        """
        self.root_dir = root_dir
        study_name = os.path.join(self.root_dir, "opt_log")
        # 枝狩りのための Pruner を作成
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=4, min_early_stopping_rate=0
        )

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=pruner,
            study_name=study_name,
            storage=f"sqlite:///{study_name}.db",
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(self.objective, n_trials=self.max_trial_num)
        optuna.visualization.plot_optimization_history(study)

        # ベストなパラメタの組み合わせの保存
        df = study.trials_dataframe()
        csv_save_pth = os.path.join(self.root_dir, "optuna_lgb.csv")
        df.to_csv(csv_save_pth)

    def objective(self, trial):
        # REF: https://qiita.com/koshian2/items/ef9c0c74fe38739599d5
        # 試行に UUID を設定
        trial_uuid = str(uuid.uuid4())
        trial.set_user_attr("uuid", trial_uuid)

        # ----------------------------------------
        # 最適化するパラメタ群
        # ----------------------------------------
        params = {
            "optimizer_name": trial.suggest_categorical(
                "optimizer", ["adam", "radam", "sgd"]
            ),
            "lr": trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
            "replaced_layer_num": trial.suggest_categorical(
                "replaced_layer_num", [1, 2, 3]
            ),
        }
        # ----------------------------------------

        self.config.optim = params["optimizer_name"]
        self.config.train.lr = params["lr"]
        self.config.replaced_layer_num = params["replaced_layer_num"]

        # <<< 訓練保存用のディレクトリの作成 <<<
        dir_name = self.config.task + f"_{self.trial_count}"
        train_dir = OneTrainDir(self.root_dir, dir_name=dir_name)

        # <<< コンフィグの初期化 <<<
        self.config.model_dir = self.mdl_dir
        self.config.record_dir = self.rec_dir
        self.config.result_dir = self.res_dir

        [mdl_dir, rec_dir, res_dir] = OneTrainLogDir(self.config, train_dir)

        # <<< コンフィグの更新 <<<
        self.config.model_dir = mdl_dir
        self.config.record_dir = rec_dir
        self.config.result_dir = res_dir

        # self.network = make_network(self.config)
        self.network = make_network(
            model_name=self.config.model,
            num_classes=self.config.num_classes,
            network_name=self.config.network,
            encoder_name=self.config.encoder_name,
            replaced_layer_num=self.config.replaced_layer_num,
            train_type=self.config.train_type,
        )
        self.trainer = make_trainer(
            self.config,
            self.network,
            device_name="auto",
        )
        self.recorder = make_recorder(self.config)

        # ネットワークの可視化
        # NetVisualization(
        #     network=self.network,
        #     recorder=self.recorder,
        #     in_width=cfg.img_width,
        #     in_height=cfg.img_height,
        # )

        # optuna が選択した最適化関数名と学習率を基に，最適化関数を読みだす
        self.optimizer = make_optimizer(self.config, self.network)
        self.scheduler = make_lr_scheduler(self.config, self.optimizer)

        # ----------#
        # Training #
        # ----------#
        print(f"{self.trial_count} 番目の訓練を実行します．")
        for epoch in range(self.config.train.epoch):
            self.recorder.epoch = epoch
            self.trainer.train(epoch, self.train_loader, self.optimizer, self.recorder)
            self.scheduler.step()

            # 訓練途中のモデルを保存する
            if (epoch + 1) % self.config.save_ep == 0:
                save_model(
                    self.network,
                    self.optimizer,
                    self.scheduler,
                    self.recorder,
                    epoch + 1,
                    self.config.model_dir,
                )

            # 検証
            if (epoch + 1) % self.config.eval_ep == 0:
                val_loss = self.trainer.val(
                    epoch, self.val_loader, recorder=self.recorder
                )
                trial.set_user_attr("val_loss", val_loss.item())

        # 訓練終了後のモデルを保存
        save_model(
            self.network,
            self.optimizer,
            self.scheduler,
            self.recorder,
            epoch,
            self.config.model_dir,
        )

        # Cfg 情報の保存
        CfgSave(self.config, train_dir)

        # -------- #
        # Testing #
        # -------- #
        print("Test を実行します．")
        test(self.config)

        self.trial_count += 1

        # 不要なオブジェクトを削除
        del self.network, self.trainer, self.optimizer, self.scheduler
        gc.collect()
        torch.cuda.empty_cache()

        return val_loss.item()


def main(config, root_dir: str = "."):

    # 訓練
    opt = OptunaTrain(config)
    opt.Train(root_dir=root_dir)


if __name__ == "__main__":
    import traceback

    debug = False
    torch.cuda.empty_cache()

    if not debug:
        try:
            main(cfg)
        except Exception as e:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    else:
        print("訓練をデバッグモードで実行します．")
        from yacs.config import CfgNode as CN
        from lib.utils.base_utils import DirCheckAndMake

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
        conf.result_dir = "result"
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
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

        """
        conf = CN()
        conf.cls_names = ["laptop", "tv"]
        conf.task = "semantic_segm"
        conf.network = "smp"
        conf.model = "unetpp"
        conf.encoder_name = "resnet18"
        conf.model_dir = "model"
        conf.record_dir = "record"
        conf.train_type = "transfer"  # or scratch
        # self.config.train_type = "scratch"
        conf.img_width = 224
        conf.img_height = 224
        conf.resume = True  # 追加学習するか
        conf.use_amp = False
        conf.ep_iter = -1
        conf.save_ep = 5
        conf.eval_ep = 1
        conf.train = CN()
        conf.train.epoch = 100
        # self.config.train.dataset = "SampleTrain"
        # self.config.train.dataset = "Sample_2Train"
        # self.config.train.dataset = "BrakeRotorsTrain"
        # self.config.train.dataset = "LinemodTrain"
        conf.train.dataset = "COCO2017Val"
        conf.train.batch_size = 20
        conf.train.num_workers = 2
        conf.train.batch_sampler = ""
        conf.train.optim = "adam"
        conf.train.criterion = ""
        conf.train.lr = 1e-3
        conf.train.scheduler = "step_lr"
        conf.train.weight_decay = 0.0
        conf.train.milestones = (20, 40, 60, 80, 100, 120, 160, 180, 200, 220)
        conf.train.warp_iter = 50
        conf.train.gamma = 0.5
        conf.train.metrics = "iou"
        conf.test = CN()
        # self.config.test.dataset = "SampleTest"
        # self.config.test.dataset = "Sample_2Test"
        # self.config.test.dataset = "LinemodTest"
        conf.test.dataset = "COCO2017Val"
        conf.test.batch_size = 20
        conf.test.num_workers = 2
        conf.test.batch_sampler = ""
        """

        dir_pth = os.path.join(pth.DATA_DIR, "trained")
        dir_pth = DirCheckAndMake(dir_pth)

        torch.cuda.empty_cache()
        try:
            main(conf, dir_pth)
        except:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
