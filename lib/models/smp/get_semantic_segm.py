import sys

sys.path.append(".")
sys.path.append("../../../")

from yacs.config import CfgNode

from lib.models.smp.unetpp import GetUNetPP as get_unet_pp

_network_factory = {"unetpp": get_unet_pp}


def GetSemanticSegm(cfg: CfgNode):
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


if __name__ == "__main__":
    from lib.datasets.make_datasets import make_data_loader

    cfg = CfgNode()
    cfg.task = "semantic_segm"
    cfg.cls_names = ["laptop", "tv"]
    cfg.num_classes = len(cfg.cls_names)
    cfg.img_width = 224
    cfg.img_height = 224
    cfg.task = "semantic_segm"
    cfg.network = "smp"
    cfg.model = "unetpp"
    cfg.encoder_name = "resnet18"
    cfg.train = CfgNode()
    cfg.train.dataset = "COCO2017Val"
    cfg.train.batch_size = 4
    cfg.train.num_workers = 2
    cfg.train.batch_sampler = ""

    model = GetSemanticSegm(cfg)

    dloader = make_data_loader(cfg, is_train=True)
    for iter, batch in enumerate(dloader):
        img = batch["img"].float()
        output = model.forward(img)
