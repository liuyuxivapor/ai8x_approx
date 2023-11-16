import torch
import torch.nn as nn
from torchaudio.functional import create_dct

TORCHAUDIO_SQUEEZER = 14

class DCT4_TORCHAUDIO(nn.Module):
    def __init__(self):
        super().__init__()
    
        # self.dct = ai8x.FusedConv1dAbs(128, 128, 1, stride=1, padding=0, bias=False, **kwargs)
        self.dct = nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        
        dct_coefs = create_dct(n_mfcc=128, n_mels=128, norm=None)
        with torch.no_grad():
            self.dct.weight = nn.Parameter(dct_coefs.transpose(0,1)[0:128,:].unsqueeze(2)/TORCHAUDIO_SQUEEZER, requires_grad=False)

    def forward(self, x):
        return self.dct(x)