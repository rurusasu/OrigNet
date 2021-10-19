import torch.nn as nn
import pytorch_lightning as pl


# 学習モデル
class Net(nn.Module):
    def __init__(
        self, input_size=784, hidden_size=100, output_size=10, batch_size=256
    ) -> None:
        super(Net, self).__init__()
        self.conv = nn.Conv2d()
