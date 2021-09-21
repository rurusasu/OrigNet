import importlib.machinery
import importlib.util
import os
import sys

sys.path.append(".")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.train.trainers.trainer import Trainer
from lib.config.config import pth


def _wrapper_factory(cfg: CfgNode):
    path = os.path.join(pth.LIB_DIR, "train", "trainers", cfg.task + ".py")
    spec = importlib.util.spec_from_file_location("NetworkWrapper", path)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    return my_module.NetworkWrapper


def make_trainer(cfg: CfgNode, network):
    module = _wrapper_factory(cfg)
    network = module(cfg, network)
    return Trainer(network)


if __name__ == "__main__":
    import sys

    sys.path.append("../../../")

    from lib.models.make_network import make_network

    cfg = CfgNode()
    cfg.task = "classify"
    cfg.model = "res_18"
    cfg.train = CfgNode()
    cfg.train.pretrained = False
    model = make_network(cfg)

    trainer = make_trainer(cfg, model)
    print(trainer)
