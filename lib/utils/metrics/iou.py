import numpy as np
import torch


class iou_score(object):
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-5):
        if 0 <= threshold <= 1:
            self.threshold = threshold
        else:
            raise ValueError(f"Invalid value for threshold: {threshold}")
        self.smooth = smooth

    def calc(self, output: np.ndarray, target: np.ndarray):
        if torch.is_tensor(output):
            output = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        output_ = output > self.threshold
        target_ = target > self.threshold
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()

        return (intersection + self.smooth) / (union + self.smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
