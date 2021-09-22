import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize(**imgs: torch.Tensor):
    """Plot images in one row."""

    n = len(imgs)
    plt.figure(figsize=(16, 5))
    for i, (name, img) in enumerate(imgs.items()):
        # img = (img.to("cpu").detach().numpy().transpose(1, 2, 0)).astype(np.uint8)
        img = img.numpy().copy().transpose(1, 2, 0)
        # img = cv2.resize(img, (w, h))
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title("".join(name.split("_")).title())
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    import sys

    sys.path.append("../../")

    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_dataset

    cfg = CN()
    cfg.task = "semantic_segm"
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CN()
    cfg.train.dataset = "LinemodTrain"

    dataset = make_dataset(cfg, cfg.train.dataset)
    img, msk = dataset[4]["img"], dataset[4]["msk"]
    visualize(imgs=img, msk=msk)
