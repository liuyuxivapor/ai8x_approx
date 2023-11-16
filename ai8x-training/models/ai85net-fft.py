import torch
import torch.nn as nn

import ai8x

class AI85Net_FFT(nn.Module):
    def __init__(self, bias=False, **kwargs):
        super().__init__()
        
        length = 64

        self.linear1 = ai8x.Linear(1 * length, 4 * length)
        self.linear2 = ai8x.Linear(4 * length, 4 * length)
        self.linear3 = ai8x.Linear(4 * length, 2 * length)
            
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

def ai85net_fft(pretrained=False, **kwargs):
    assert not pretrained
    return AI85Net_FFT(**kwargs)

models = [
    {
        'name': 'ai85net_fft',
        'min_input': 64,
        'dim': 1,
    },
]