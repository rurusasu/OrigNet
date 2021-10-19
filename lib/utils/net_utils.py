import os

import torch
import torch.nn as nn


def load_model(
    network,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    recorder,
    model_dir: str,
    resume: bool = True,
    epoch: int = -1,
):
    """
    事前学習により保存されたモデルを読みだす関数

    Args:
        network()
    """
    if not resume:
        os.system("rm -rf {}".format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print("Load model: {}".format(os.path.join(model_dir, "{}.pth".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    network.load_state_dict(pretrained_model["net"])
    optimizer.load_state_dict(pretrained_model["optim"])
    scheduler.load_state_dict(pretrained_model["scheduler"])
    recorder.load_state_dict(pretrained_model["recorder"])
    return pretrained_model["epoch"] + 1


def load_network(
    network, model_dir, resume: bool = True, epoch: int = -1, strict: bool = True
):
    """ """
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pth" in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print("Load model: {}".format(os.path.join(model_dir, "{}.pth".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    network.load_state_dict(pretrained_model["net"], strict=strict)

    return pretrained_model["epoch"] + 1


def save_model(
    network,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    recorder,
    epoch: int,
    model_dir: str,
):
    """
    訓練されたモデルを保存する関数
    """
    os.system("mkdir -p {}".format(model_dir))
    torch.save(
        {
            "net": network.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "recorder": recorder.state_dict(),
            "epoch": epoch,
        },
        os.path.join(model_dir, "{}.pth".format(epoch)),
    )

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:
        return
    os.system("rm {}".format(os.path.join(model_dir, "{}.pth".format(min(pths)))))


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size())
            )
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    # BatchNormのみFP32フォーマットにしないと性能が出ない。
    # BatchNormレイヤを検索し、このレイヤのみFP32に設定。
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    ネットワークのパラメタを半精度に変換する関数．
    以下のサイトからコピー
    REF: https://aru47.hatenablog.com/entry/2020/11/06/225942
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))
