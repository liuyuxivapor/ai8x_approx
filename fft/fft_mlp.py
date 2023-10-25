import torch
import torch.nn as nn
# from quant_ultra import *

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        # W_BIT = 4
        # A_BIT = 4
        # conv2d_q = conv2d_Q_fn(W_BIT)
        self.layers = nn.Sequential(
            nn.Linear(1, 4), nn.ReLU(),
            nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
