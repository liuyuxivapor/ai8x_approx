import torch
import torch.nn as nn

import ai8x

class AI85Net_FFT(nn.Module):
    def __init__(self, num_samples=2048, bias=False, **kwargs):
        super().__init__()

        self.linear1 = ai8x.FusedSoftwareLinearReLU(1, 4)
        self.linear2 = ai8x.FusedSoftwareLinearReLU(4, 4)
        self.linear3 = ai8x.FusedSoftwareLinearReLU(4, 2)
            
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

def ai85net_fft(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_FFT(**kwargs)

models = [
    {
        'name': 'ai85net_fft',
        'min_input': 1,
        'dim': 2,
    },
]