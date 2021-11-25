from typing import Literal

import segmentation_models_pytorch as smp

# REF: https://smp.readthedocs.io/en/latest/
# GitHub: https://github.com/qubvel/segmentation_models.pytorch


def GetUNetPP(
    encoder_name,
    num_classes,
    train_type: Literal["scratch", "transfer"] = "scratch",
    activation=None,
):
    if train_type == "transfer":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
            activation=activation,
        )
    elif train_type == "scratch":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=num_classes,
            activation=activation,
        )
    else:
        raise ValueError("For train_type, select scratch or transfer.")

    return model
