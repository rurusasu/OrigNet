from lib.config.config import cfg, pth
import torch
import torchvision
from torchvision import transforms, datasets
import os
import sys

sys.path.append('..')
sys.path.append('../..')


dataset_save_pth = pth.DATA_DIR

# 画像ファイルを読み込むための準備（channels x H x W）
transform = transforms.Compose([
    transforms.ToTensor()
])

# データセットの取得
train_val = datasets.MNIST(
    root=dataset_save_pth,
    train=True,
    download=True,
    transform=transform
)

test = datasets.MNIST(
    root=dataset_save_pth,
    train=False,
    download=True,
    transform=transform
)

# train : val = 80% : 20%
n_train = int(len(train_val) * 0.8)
n_val = len(train_val) - n_train

# データをランダムに分割
train, val = torch.utils.data.random_split(train_val, [n_train, n_val])
