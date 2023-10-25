import torch
import torch.nn as nn
from quant_ultra import *

class ultra_net(nn.Module):
    def __init__(self):
        super(ultra_net, self).__init__()
        W_BIT = 4
        A_BIT = 4
        # conv2d_q = conv2d_Q_fn(W_BIT)
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
