import sys

sys.path.append(".")
sys.path.append("../../../")

from lib.models.smp.unetpp import GetUNetPP as get_unet_pp

_network_factory = {"unetpp": get_unet_pp}


def GetSemanticSegm(cfg):
    if cfg.model not in _network_factory:
        raise ValueError(f"The specified cfg.network={cfg.model} does not exist.")

    if "encoder_name" not in cfg and "num_classes" not in cfg:
        raise ValueError(
            "The required config parameter for `GetSemanticSegm` is not set."
        )

    arch = cfg.model

    get_model = _network_factory[arch]

    model = get_model(encoder_name=cfg.encoder_name, num_classes=cfg.num_classes)

    return model
