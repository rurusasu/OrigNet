import argparse
import os
import sys

from easydict import EasyDict
from yacs.config import CfgNode as CN


_model_factory = {"original"}

"""
Path Setting
"""
pth = EasyDict()

pth.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
pth.LIB_DIR = os.path.dirname(pth.CONFIG_DIR)
pth.ROOT_DIR = os.path.dirname(pth.LIB_DIR)
pth.CONFIGS_DIR = os.path.join(pth.ROOT_DIR, "configs")

"""
Dataset Path
"""
pth.DATA_DIR = os.path.join(pth.ROOT_DIR, "data")
# pth.MNIST_DIR = os.path.join(pth.DATA_DIR, 'mnist')


def add_path():
    """システムのファイルパスを設定するための関数"""

    for key, value in pth.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()


"""
Config Setting
"""
cfg = CN()

"""
Default values setting
"""
# task
# classify or semantic_segm
cfg.task = ""
# class names
cfg.cls_names = []
# cnns, smp
cfg.network = "cnns"
# 例: alex
cfg.model = ""
# SemanticSegmentation 用のモデルで使用
# 例: resnet18
cfg.encoder_name = ""
# 例: original
cfg.model_dir = "data/model"
# recorder
cfg.record_dir = "data/record"
# result
cfg.result_dir = "data/result"
# transfer or scratch
cfg.train_type = "transfer"
# img_size
cfg.img_width = 255
cfg.img_height = 255
# 追加学習するか
cfg.resume = True
# 余ったバッチを切り捨てるか
cfg.ep_iter = -1
# 何 epoch で保存するか
cfg.save_ep = 5
cfg.eval_ep = 1
# gpus
cfg.gpus = [0, 1, 2, 3]
# use amp
cfg.use_amp = True

# ---------------------
# train
# ---------------------
cfg.train = CN()

# dataset
cfg.train.dataset = ""
cfg.train.epoch = 140
cfg.train.batch_size = 4
cfg.train.num_workers = 2
# image_size
cfg.train.batch_sampler = ""
cfg.train.optim = "adam"
cfg.train.criterion = ""
cfg.train.lr = 1e-3
cfg.train.scheduler = "step_lr"
cfg.train.weight_decay = 0.0
cfg.train.milestones = (20, 40, 60, 80, 100, 120, 160, 180, 200, 220)
cfg.train.warp_iter = 5
cfg.train.gamma = 0.5
cfg.train.metrics = "iou"

cfg.test = CN()
cfg.test.dataset = ""
cfg.test.batch_size = 20
cfg.test.num_workers = 2
# image_size
cfg.test.batch_sampler = ""


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError("task must be specified")

    # assign the gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])

    cfg.model_dir = os.path.join(cfg.task, cfg.train.dataset, cfg.model, cfg.model_dir)
    cfg.record_dir = os.path.join(
        cfg.task, cfg.train.dataset, cfg.model, cfg.record_dir
    )
    cfg.result_dir = os.path.join(cfg.task, cfg.train.dataset, cfg.model, "result")


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    opts_idx = [
        i
        for i in range(0, len(args.opts), 2)
        if args.opts[i].split(".")[0] in cfg.keys()
    ]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    cfg.merge_from_list(opts)
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg_file", default=os.path.join(pth.CONFIGS_DIR, "default.yaml"), type=str
)
parser.add_argument("--test", action="store_true", dest="test", default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument("--det", type=str, default="")
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
