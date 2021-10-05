import sys

sys.path.append(".")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.train.trainers.classify import ClassifyNetworkWrapper
from lib.train.trainers.semantic_segm import SemanticSegmentationNetworkWrapper
from lib.train.trainers.trainer import Trainer


_wrapper_factory = {
    "classify": ClassifyNetworkWrapper,
    "semantic_segm": SemanticSegmentationNetworkWrapper,
}


def make_trainer(cfg: CfgNode, network, device: str = "cpu"):
    """
    Trainer クラスを呼び出す関数
    device 引数について不明な場合は以下を参照．
    REF: https://note.nkmk.me/python-pytorch-device-to-cuda-cpu/

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        network: 訓練されるネットワーク
        device(str): 'cpu' もしくは 'cuda: n' ここで n はGPU 番号．Default to 'cpu'.
    """
    wrapper = _wrapper_factory[cfg.task]
    network = wrapper(cfg, network, device)
    return Trainer(network, device)


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
