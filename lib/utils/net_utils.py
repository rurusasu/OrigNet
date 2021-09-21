import os

import torch


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
