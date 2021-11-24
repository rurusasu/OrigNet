import torch
import segmentation_models_pytorch as smp


class DiceLoss(smp.utils.losses.DiceLoss):
    """入力が Nan もしくは inf の場合，それらを 0 に置き換えて入力するように変更した DiceLoss．

    Args:
        smp ([type]): [description]
    """

    def __init__(self):
        super().__init__()
        self.bce = smp.utils.losses.DiceLoss()

    def forward(self, input, target):
        input = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        input = torch.where(torch.isinf(input), torch.zeros_like(input), input)
        input = torch.where(input > 1, torch.ones_like(input), input)  # 1を超える場合には1にする

        target = target.float()

        return self.bce(input, target)
