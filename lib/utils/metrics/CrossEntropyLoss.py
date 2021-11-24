import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, input, target):
        input = torch.where(torch.isnan(input), torch.zeros_like(input), input)
        input = torch.where(torch.isinf(input), torch.zeros_like(input), input)
        input = torch.where(input > 1, torch.ones_like(input), input)  # 1を超える場合には1にする

        target = target.long()

        return self.cel(input, target)
