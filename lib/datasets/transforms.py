import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from yacs.config import CfgNode


# class ToTensor(object):
#    def __call__(self, img: torch.Tensor) -> torch.Tensor:
#        # return np.asarray(img).astype(np.float32) / 255.0
#        return img / 255.0


def make_transforms(cfg: CfgNode, is_train: bool) -> transforms:
    """データ拡張に使用する Transforms を作成する関数
    Args:
        cfg (CfgNode): データセット名などのコンフィグ情報
        is_train (bool): 訓練用データセットか否か．

    Return:
        (transforms): データ変換に使用する関数群
    """
    if is_train is True:
        transform = Compose(
            [
                # Tensor型に変換する
                ToTensor(),
                # 画像をimg_width×img_heightの大きさに統一する
                transforms.Resize((cfg.img_width, cfg.img_height)),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # Tensor型に変換する
                ToTensor(),
                # 画像をimg_width×img_heightの大きさに統一する
                transforms.Resize((cfg.img_width, cfg.img_height)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform
