import numpy as np
import torch


def iou_score(
    pred_data: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> torch.Tensor:
    if 0 <= threshold <= 1:
        pass
    else:
        raise ValueError(f"Invalid value for threshold: {threshold}")

    if torch.is_tensor(pred_data):
        pred_data = torch.sigmoid(pred_data).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    pred_data_ = pred_data > threshold
    target_ = target > threshold
    intersection = (pred_data_ & target_).sum()
    union = (pred_data_ | target_).sum()

    iou = np.array((intersection + smooth) / (union + smooth), dtype=np.float32)
    iou = torch.from_numpy(iou.astype(np.float32)).clone()
    return iou


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
