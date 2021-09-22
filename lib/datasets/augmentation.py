import math
import random
from typing import Dict

import cv2
import numpy as np
from PIL import Image


def crop_or_padding_to_fixed_size(img, th, tw):
    h, w, _ = img.shape
    hpad, wpad = th >= h, tw >= w

    hbeg = 0 if hpad else np.random.randint(0, h - th)
    wbeg = (
        0 if wpad else np.random.randint(0, w - tw)
    )  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    hend = hbeg + th
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg : hbeg + nh, wbeg : wbeg + nw] = img

        img = new_img

    return img


def crop_or_padding(img, hratio, wratio):
    """
    if ratio<1.0 then crop, else padding
    :param img:
    :param hratio:
    :param wratio:
    :return:
    """
    h, w, _ = img.shape
    hd = int(hratio * h - h)
    wd = int(wratio * w - w)
    hpad = hd > 0
    wpad = wd > 0

    if hpad:
        ohbeg = hd // 2
        ihbeg = 0
        hlen = h
    else:
        ohbeg = 0
        ihbeg = -hd // 2
        hlen = h + hd

    if wpad:
        owbeg = wd // 2
        iwbeg = 0
        wlen = w
    else:
        owbeg = 0
        iwbeg = -wd // 2
        wlen = w + wd

    out_img = np.zeros([h + hd, w + wd, 3], np.uint8)
    out_img[ohbeg : ohbeg + hlen, owbeg : owbeg + wlen] = img[
        ihbeg : ihbeg + hlen, iwbeg : iwbeg + wlen
    ]

    return out_img


def resize_keep_aspect_ratio(img, imsize, intp_type=cv2.INTER_LINEAR):
    h, w = img.shape[0], img.shape[1]
    ratio = imsize / max(h, w)
    hbeg, wbeg = 0, 0
    # padding_mask=np.zeros([imsize,imsize],np.uint8)
    if h > w:
        hnew = imsize
        wnew = int(ratio * w)
        img = cv2.resize(img, (wnew, hnew), interpolation=intp_type)
        if wnew < imsize:
            if len(img.shape) == 3:
                img_pad = np.zeros([imsize, imsize, img.shape[2]], img.dtype)
            else:
                img_pad = np.zeros([imsize, imsize], img.dtype)
            wbeg = int((imsize - wnew) / 2)
            img_pad[:, wbeg : wbeg + wnew] = img

            # padding_mask[:,:wbeg]=1
            # padding_mask[:,wbeg+wnew:]=1
            img = img_pad
    else:
        hnew = int(ratio * h)
        wnew = imsize
        img = cv2.resize(img, (wnew, hnew), interpolation=intp_type)
        if hnew < imsize:
            if len(img.shape) == 3:
                img_pad = np.zeros([imsize, imsize, img.shape[2]], img.dtype)
            else:
                img_pad = np.zeros([imsize, imsize], img.dtype)
            hbeg = int((imsize - hnew) / 2)
            img_pad[hbeg : hbeg + hnew, :] = img

            # padding_mask[:,:hbeg]=1
            # padding_mask[:,hbeg+hnew:]=1
            img = img_pad

    # x_new=x_ori*ratio+wbeg
    # y_new=y_ori*ratio+hbeg
    return img, ratio, hbeg, wbeg


# <<< higher level api <<<
def resize_with_crop_or_pad_to_fixed_size(img, ratio):
    h, w, _ = img.shape
    th, tw = int(math.ceil(h * ratio)), int(math.ceil(w * ratio))
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)

    if ratio > 1.0:
        # crop
        hbeg, wbeg = np.random.randint(0, th - h), np.random.randint(0, tw - w)
        result_img = img[hbeg : hbeg + h, wbeg : wbeg + w]
    else:
        # padding
        result_img = np.zeros([h, w, img.shape[2]], img.dtype)
        hbeg, wbeg = (h - th) // 2, (w - tw) // 2
        result_img[hbeg : hbeg + th, wbeg : wbeg + tw] = img

    return result_img


def rotate(img, rot_ang_min, rot_ang_max):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    img = cv2.warpAffine(
        img,
        R,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return img


def rotate_instance(img: np.ndarray, msk: np.ndarray, rot_ang_min, rot_ang_max):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    nonzero = np.nonzero(msk)
    hs, ws = np.mean(nonzero[0]), np.mean(nonzero[1])
    # 回転行列を作る．
    # REF: https://qiita.com/mo256man/items/a32ddf5ddbaaf767d319
    # arg:
    #   center: 回転中心．
    #   angle: 回転角度
    #   scale: 倍率
    R = cv2.getRotationMatrix2D((ws, hs), degree, 1)
    msk = cv2.warpAffine(
        msk,
        R,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img = cv2.warpAffine(
        img,
        R,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return img, msk


def flip(img):
    img = np.flip(img, 1)
    return img


def blur_image(img, sigma=3):
    return cv2.GaussianBlur(img, (sigma, sigma), 0)


class RandomBlur(object):
    def __init__(self, prob) -> None:
        self.prob = prob

    def __call__(self, img) -> np.ndarray:
        img = np.asarray(img).astype(np.float32)
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            img = cv2.GaussianBlur(img, (sigma, sigma), 0)
        return img


def add_noise(image):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
    return noisy


def _augmentation(
    img: np.ndarray, msk: np.ndarray, height: int, width: int
) -> np.ndarray:
    foreground = np.sum(msk)

    if foreground > 0:
        # randomly rotate around the center of the instance
        # img = rotate(img, -30, 30)
        img, msk = rotate_instance(img, msk, -30, 30)

    # randomly crop and resize
    # 1. firstly crop a region which is [scale_min,scale_max]*[height,width],
    # which ensures that the area of the intersection between the cropped region
    # and the instance region is at least overlap_ratio**2 of instance region.
    # 2. if the region is larger than original image, then padding 0
    # 3. then resize the cropped image to [height, width]
    # (bilinear for image, nearest for mask)

    # msk 画像が存在しない場合
    else:
        img = crop_or_padding_to_fixed_size(img, height, width)

    return img, msk


def augmentation(
    imgs: Dict[Image.Image, Image.Image], height: int, width: int, split: str
) -> Dict[np.ndarray, np.ndarray]:
    """画像オーギュメンテーションを行う関数

    Args:
        imgs (Dict[img:PIL.Image, msk:PIL.Image]): 画像とそのマスク画像が保存された辞書
        height (int): リサイズ時の高さ
        width (int): リサイズ時の幅
        split (str):

    Returns:
        imgs (Dict[img: np.ndarray, msk: np.ndarray]): 画像とそのマスク画像が保存された辞書
    """
    img, msk = imgs["img"], imgs["msk"]

    if split == "train":
        img, msk = _augmentation(img, msk, height, width)

        if np.random.random() < 0.5:
            img = blur_image(img, np.random.choice([3, 5, 7, 9]))

    imgs["img"], imgs["msk"] = img, msk
    return imgs
