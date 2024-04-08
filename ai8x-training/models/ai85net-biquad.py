import torch
import torch.nn as nn

import ai8x

class AI85Net_BIQUAD(nn.Module):
    def __init__(self, bias=True, **kwargs):
        super().__init__()

        self.fc1 = ai8x.FusedLinearReLU(1, 2)
        self.fc2 = ai8x.FusedLinearReLU(2, 2)
        self.fc3 = ai8x.FusedLinearReLU(2, 16)
        self.fc4 = ai8x.Linear(16, 1)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def ai85net_biquad(pretrained=False, **kwargs):
    assert not pretrained
    return AI85Net_BIQUAD(**kwargs)

models = [
    {
        'name': 'ai85net_biquad',
        'min_input': 1,
        'dim': 1,
    },
]