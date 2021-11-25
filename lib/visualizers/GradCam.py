import sys
from typing import Literal

sys.path.append("../../")

import torch
from matplotlib import pyplot as plt
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

from lib.utils.base_utils import Tensor2Ndarray3D


class CAM(object):
    def __init__(self, model_name: str, dataset_name: str) -> None:

    def main(
        model,
        input_tensor: torch.Tensor,
        target_layers,
        cam_type: Literal["grad_cam", "grad_cam_pp"],
        use_cuda: bool = True,
    ):
        # input_tensor = torch.zeros((1, 3, 224, 224))
        rgb_img = input_tensor[1, :, :, :]
        input_tensor = rgb_img.unsqueeze(dim=0)

        rgb_img = Tensor2Ndarray3D(rgb_img)
        if cam_type == "grad_cam":
            # いったんCAMオブジェクトを構築し、その後、多くの画像上でそれを再利用
            cam = GradCAM(model, target_layers=target_layers, use_cuda=use_cuda)
        elif cam_type == "grad_cam_pp":
            cam = GradCAMPlusPlus(model, target_layers=target_layers, use_cuda=use_cuda)

        target_category = 1
        # 平滑化を適用するために、aug_smooth = Trueおよびeigen_smooth = Trueを渡すこともできます。
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
        plt.show()


if __name__ == "__main__":
    import os
    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_data_loader
    from lib.models.make_network import make_network
    from lib.utils.net_utils import load_network

    conf = CN()
    conf.task = "classify"
    conf.network = "cnns"
    conf.model = "res_18"
    conf.train_type = "scratch"
    conf.img_width = 224
    conf.img_height = 224
    conf.test = CN()
    conf.test.dataset = "SampleTest"
    conf.test.batch_size = 24
    conf.test.num_workers = 2
    conf.test.batch_sampler = ""

    dloader = make_data_loader(conf, ds_category="test")
    conf.num_classes = len(dloader.dataset.cls_names)
    batch_iter = iter(dloader)
    batch = next(batch_iter)
    img, _ = batch["img"], batch["target"]
    # img = img[1, :, :, :]

    model = make_network(conf)
    _ = load_network(
        model, "/mnt/d/My_programing/OrigNet/data/trained/1/classify_1/model/"
    )

    target_layer = model.layer4
    cam = CAM(
        model, input_tensor=img, target_layers=target_layer, cam_type="grad_cam_pp"
    )
