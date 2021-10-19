import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


def visualize(**imgs: torch.Tensor) -> plt.figure:
    """Plot images in one row."""

    n = len(imgs)
    fig = plt.figure(figsize=(16, 5))

    input_img = None
    for i, (name, img) in enumerate(imgs.items()):
        if len(img) > 0:
            # もし，画像がグレー階調もしくはRGBの場合
            img = img.detach().clone().cpu()
            if "input" in name:
                # Tensor[C, H, W] -> ndarray[H, W, C]
                img = img.numpy().transpose(1, 2, 0)
                input_img = img  # 入力画像を一旦保存しておく．
            elif ("mask" in name) or ("target" in name):  # mask の場合
                img = img.numpy()
            elif ("predict" in name) or ("output" in name):
                # それ以外の場合(画像が One Hot Label の場合)
                #  出力から最大クラスを求める．
                img = img.numpy()
                img = np.argmax(img, axis=0)
                # ndarry[H, W, C] -> PIL[W, H, C]
                img = Image.fromarray(np.uint8(img))

                # もし，既に入力画像が保存されている場合
                if input_img is not None:
                    input_img = Image.fromarray(np.uint8(input_img * 255.0))
                    trans_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    img = img.convert("RGBA")

                    for x in range(img.size[0]):
                        for y in range(img.size[1]):
                            # 推論結果画像のピクセルデータを取得
                            pixel = img.getpixel((x, y))
                            r, g, b, _ = pixel

                            # pixcel value(0, 0, 0) の背景ならそのままにして透過させる．
                            if (r == 0) and (g == 0) and (b == 0):
                                continue
                            else:
                                # それ以外の色は用意した画像にピクセルを書き込む
                                trans_img.putpixel((x, y), (r, g, b, 150))
                                # 150 は透過度の大きさを指定している．

                    img = Image.alpha_composite(input_img.convert("RGBA"), trans_img)

            # img = cv2.resize(img, (w, h))
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title("".join(name.split("_")).title())
            plt.imshow(img)

    return fig


def visualize_np(**imgs: np.ndarray) -> plt.figure:
    """Plot images in one row."""

    n = len(imgs)
    fig = plt.figure(figsize=(16, 5))

    input_img = None
    for i, (name, img) in enumerate(imgs.items()):
        if len(img) > 0:
            # もし，画像がグレー階調もしくはRGBの場合
            if "input" in name:
                # Tensor[C, H, W] -> ndarray[H, W, C]
                # img = img.numpy().transpose(1, 2, 0)
                input_img = img  # 入力画像を一旦保存しておく
            elif ("predict" in name) or ("output" in name):
                # それ以外の場合(画像が One Hot Label の場合)
                #  出力から最大クラスを求める．
                img = np.argmax(img, axis=0)
                # ndarry[H, W, C] -> PIL[W, H, C]
                img = Image.fromarray(np.uint8(img))

                # もし，既に入力画像が保存されている場合
                if input_img is not None:
                    input_img = Image.fromarray(np.uint8(input_img * 255.0))
                    trans_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    img = img.convert("RGBA")

                    for x in range(img.size[0]):
                        for y in range(img.size[1]):
                            # 推論結果画像のピクセルデータを取得
                            pixel = img.getpixel((x, y))
                            r, g, b, _ = pixel

                            # pixcel value(0, 0, 0) の背景ならそのままにして透過させる．
                            if (r == 0) and (g == 0) and (b == 0):
                                continue
                            else:
                                # それ以外の色は用意した画像にピクセルを書き込む
                                trans_img.putpixel((x, y), (r, g, b, 150))
                                # 150 は透過度の大きさを指定している．

                    img = Image.alpha_composite(input_img.convert("RGBA"), trans_img)

            # img = cv2.resize(img, (w, h))
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title("".join(name.split("_")).title())
            plt.imshow(img)

    return fig


if __name__ == "__main__":
    import sys

    sys.path.append("../../")

    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_dataset

    cfg = CN()
    # cfg.task = "classify"
    cfg.task = "semantic_segm"
    cfg.img_width = 200
    cfg.img_height = 200
    cfg.train = CN()
    # cfg.train.dataset = "SampleTrain"
    cfg.train.dataset = "LinemodTrain"

    dataset = make_dataset(cfg, cfg.train.dataset)
    img, msk = dataset[4]["img"], dataset[4]["msk"]
    visualize(imgs=img, msk=msk)
