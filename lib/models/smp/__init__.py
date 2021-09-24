import sys

sys.path.append(".")
sys.path.append("../../../")

from lib.models.smp.unetpp import get_model


def get_network(cfg):
    model = get_model(encoder_name=cfg.model, num_classes=cfg.num_classes)

    return model
