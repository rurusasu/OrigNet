import sys
from typing import Literal

sys.path.append(".")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.train.trainers.classify import ClassifyNetworkWrapper
from lib.train.trainers.semantic_segm import SemanticSegmentationNetworkWrapper
from lib.train.trainers.trainer import Trainer
from lib.utils.net_utils import network_to_half


_wrapper_factory = {
    "classify": ClassifyNetworkWrapper,
    "semantic_segm": SemanticSegmentationNetworkWrapper,
}


def make_trainer(
    cfg: CfgNode, network, device_name: Literal["cpu", "cuda", "auto"] = "auto"
):
    """
    ネットワークを半精度で訓練する場合，そのパラメタを fp16 に変換する．
    ネットワークの出力から損失関数の計算部分までをラッピングする．
    Trainer クラスを呼び出す．
    device 引数について不明な場合は以下を参照．
    REF: https://note.nkmk.me/python-pytorch-device-to-cuda-cpu/

    Args:
        cfg (CfgNode): `config` 情報が保存された辞書．
        network: 訓練されるネットワーク
        device(str): 'cpu' もしくは 'cuda: n' ここで n はGPU 番号．Default to 'cpu'.
    """
    network = network_to_half(network) if cfg.use_amp else network
    wrapper = _wrapper_factory[cfg.task]
    network = wrapper(cfg, network)
    return Trainer(network, device_name=device_name, use_amp=cfg.use_amp)


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
