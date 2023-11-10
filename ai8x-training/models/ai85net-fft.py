import torch
import torch.nn as nn

import ai8x

class AI84Net_FFT(nn.Module):
    def __init__(self, bias=False, **kwargs):
        super().__init__()

        self.linear1 = ai8x.FusedSoftwareLinearReLU(1, 4)
        self.linear2 = ai8x.FusedSoftwareLinearReLU(4, 4)
        self.linear3 = ai8x.FusedSoftwareLinearReLU(4, 2)
            
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

def ai84net_fft(pretrained=False, **kwargs):
    """
    Constructs a AI84Net5 model.
    """
    assert not pretrained
    return AI84Net_FFT(**kwargs)

models = [
    {
        'name': 'ai84net_fft',
        'min_input': 1,
        'dim': 2,
    },
]