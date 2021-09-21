import importlib.machinery
import importlib.util
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

from yacs.config import CfgNode

from lib.config.config import pth


def make_network(cfg: CfgNode):
    """Network を読みだす関数

    Args:
        cfg (CfgNode):

    Returns:
        [type]: [description]
    """
    if "train" not in cfg:
        raise ("The required parameter for `make_network` is not set.")
    name = cfg.model
    if name.find("_"):
        name = name[: name.find("_")]
    path = os.path.join(pth.LIB_DIR, "models", name, "__init__.py")
    # spec = importlib.util.spec_from_file_location("get_network", path)
    spec = importlib.util.spec_from_file_location("__init__", path)
    get_network_fun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(get_network_fun)
    network = get_network_fun.get_network(cfg)
    return network


if __name__ == "__main__":
    cfg = CfgNode()
    cfg.model = "res_152"
    cfg.train = CfgNode()
    cfg.train.pretrained = False

    network = make_network(cfg)
    print(network)
