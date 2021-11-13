import sys

sys.path.append("../../")

import torch

from lib.utils.base_utils import SelectDevice


def NetVisualization(network, recorder, in_width, in_height):
    device_name, num_devices = SelectDevice()
    device = torch.device(device_name)
    # torch.tensor([C, H, W])
    input = torch.zeros((1, 3, in_width, in_height), dtype=torch.float32, device=device)
    recorder.VisualizeNetwork(network, input)
    del input
