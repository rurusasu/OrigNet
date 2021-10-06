# from torchvision import transforms
from torchvision import transforms
import albumentations as albu
from yacs.config import CfgNode


# def to_tensor(x, **kwargs):
#    x = torch.from_numpy(x.astype(np.float32)).clone()
# x.transpose(2, 0, 1).astype("float32")
#    return x

# albu_transforms = albu.Compose(
#     [
#         albu.HorizontalFlip(p=0.5),
#         albu.RandomBrightnessContrast(p=0.2),
#     ]
# )


# def albumentations_transform(image, transform=albu_transforms):
#     if transform:
#         augmented = transform(image=image)
#         image = augmented["image"]
#     return image


def make_transforms(cfg: CfgNode, is_train: bool) -> albu.Compose:
    """データ拡張に使用する Transforms を作成する関数
    Args:
        cfg (CfgNode): データセット名などのコンフィグ情報
        is_train (bool): 訓練用データセットか否か．

    Return:
        (transforms): データ変換に使用する関数群
    """
    if is_train is True:
        transform = [
            # Tensor型に変換する
            transforms.ToTensor(),
            # transforms.Lambda(albumentations_transform),
        ]
    else:
        transform = [
            # Tensor型に変換する
            transforms.ToTensor(),
        ]

    return transforms.Compose(transform)
