import os
import sys
from typing import Dict, Literal

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
from tqdm import tqdm

from lib.utils.base_utils import DirCheckAndMake, Tensor2Ndarray3D


class CAM(object):
    def __init__(
        self,
        network: torch.nn,
        target_layers: torch.nn,
        cam_type: Literal["grad_cam", "grad_cam_pp"] = "grad_cam",
        use_cuda: bool = True,
    ) -> None:
        """様々なタイプの ネットワークの判断根拠可視化手法を選択可能なクラス．

        Args:
            network (torch.nn): 可視化したいネットワークの構造
            target_layers (torch.nn): 可視化したい層
            cam_type (Literal[, optional): 可視化に用いるアルゴリズム. Defaults to "grad_cam".
            use_cuda (bool, optional): 計算に cuda を用いる．Defaults to True.
        """
        self.cam_type = cam_type
        if self.cam_type == "grad_cam":
            # いったんCAMオブジェクトを構築し、その後、多くの画像上でそれを再利用
            self.cam = GradCAM(network, target_layers=target_layers, use_cuda=use_cuda)
        elif self.cam_type == "grad_cam_pp":
            self.cam = GradCAMPlusPlus(
                network, target_layers=target_layers, use_cuda=use_cuda
            )

    def main(
        self,
        data_loader: torch.utils.data.DataLoader,
        draw_fig: bool = False,
        fig_save_path: str = "./",
    ):
        """判断根拠可視化を行うクラスのメイン関数．

        Args:
            data_loader (torch.utils.data.DataLoader): 入力データ作成用の DataLoader.
            draw_fig (bool, optional): 可視化結果を表示する. Defaults to False.
            fig_save_path (str, optional): 可視化結果を保存するディレクトリ. Defaults to "./".
        """
        fig_save_path = os.path.join(fig_save_path, self.cam_type)
        fig_save_path = DirCheckAndMake(fig_save_path)
        fig_save_path = os.path.join(fig_save_path, "{}.png")

        with tqdm(total=len(data_loader), leave=True, desc="train") as pbar:
            for iteration, batch in enumerate(data_loader):
                input, target = batch["img"], batch["target"]
                for i in range(input.shape[0]):
                    rgb_img = input[i, :, :, :]
                    input_tensor = rgb_img.unsqueeze(dim=0)
                    rgb_img = Tensor2Ndarray3D(rgb_img)

                    target_category = target[i].item()
                    # 平滑化を適用するために、aug_smooth = Trueおよびeigen_smooth = Trueを渡すこともできます。
                    grayscale_cam = self.cam(
                        input_tensor=input_tensor, target_category=target_category
                    )

                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(
                        rgb_img, grayscale_cam, use_rgb=True
                    )
                    plt.imshow(visualization)
                    plt.axis("off")
                    if draw_fig:
                        plt.show()
                    plt.savefig(fig_save_path.format(str(i)))
                pbar.update()


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    from lib.datasets.make_datasets import make_data_loader
    from lib.models.make_network import make_network
    from lib.utils.net_utils import load_network

    conf = CN()
    conf.task = "classify"
    conf.network = "cnns"
    conf.model = "res_18"
    conf.encoder_name = ""
    conf.train_type = "scratch"
    conf.img_width = 224
    conf.img_height = 224
    conf.max_iter = 1
    conf.test = CN()
    conf.test.dataset = "SampleTest"
    conf.test.batch_size = 24
    conf.test.num_workers = 2
    conf.test.batch_sampler = ""
    """
    dloader = make_data_loader(
        dataset_name=conf.test.dataset,
        batch_size=conf.test.batch_size,
        batch_sampler=conf.test.batch_sampler,
        ds_category="test",
        img_shape={"width": conf.img_width, "height": conf.img_height},
        num_workers=conf.test.num_workers,
        task=conf.task,
    )
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
    cam = main(
        model, input_tensor=img, target_layers=target_layer, cam_type="grad_cam_pp"
    )
    """
    data_loader = make_data_loader(
        dataset_name=conf.test.dataset,
        batch_size=conf.test.batch_size,
        batch_sampler=None,
        ds_category="test",
        img_shape={"width": conf.img_width, "height": conf.img_height},
        is_distributed=False,
        max_iter=conf.max_iter,
        normalization=False,
        num_workers=conf.test.num_workers,
        task=conf.task,
        toTensor=True,
    )

    num_classes = len(data_loader.dataset.cls_names)

    network = make_network(
        model_name=conf.model,
        num_classes=num_classes,
        network_name=conf.network,
        encoder_name=conf.encoder_name,
    )

    _ = load_network(
        network=network,
        model_dir="/mnt/d/My_programing/OrigNet/data/trained/1/classify_1/model/",
    )

    target_layers = network.layer4

    fig_save_path = "/mnt/d/My_programing/OrigNet/data/trained/1/classify_1/result"

    cam = CAM(network=network, target_layers=target_layers, cam_type="grad_cam")
    cam.main(data_loader=data_loader, fig_save_path=fig_save_path)
