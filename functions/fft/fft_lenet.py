import torch
import torch.nn as nn
from quant_ultra import *

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        W_BIT = 4
        A_BIT = 4
        # conv2d_q = conv2d_Q_fn(W_BIT)
        self.layers = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            # nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),
            # nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Flatten(),            
            nn.Linear(16 * 64, 512), nn.Sigmoid(),
            nn.Linear(512, 64), nn.Sigmoid(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
