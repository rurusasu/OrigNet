import segmentation_models_pytorch as smp

# REF: https://smp.readthedocs.io/en/latest/
# GitHub: https://github.com/qubvel/segmentation_models.pytorch


def get_model(encoder_name, num_classes, activation=None):
    if activation is not None:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
            activation=activation,
        )
    else:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            classes=num_classes,
        )

    return model
