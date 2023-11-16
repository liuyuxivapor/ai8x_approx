import torch
import torch.nn as nn
import ai8x
from torchaudio.functional import create_dct

TORCHAUDIO_SQUEEZER = 14

# class AI85Net_DCT4(nn.Module):
#     def __init__(self, num_channels=1, length=512, bias=False, **kwargs):
#         super().__init__()
    
#         self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.bn1 = nn.BatchNorm1d(8)

#         self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.bn2 = nn.BatchNorm1d(16)

#         self.conv3 = nn.Conv1d(16, 8, kernel_size=1, stride=1, padding=0, bias=bias)
        
#         self.flat = nn.Flatten()
#         self.fc = nn.Linear(512 * 8, 512)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d or nn.BatchNorm1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         relu = nn.ReLU()
#         x = relu(self.bn1(self.conv1(x)))
#         x = relu(self.bn2(self.conv2(x)))
#         x = self.conv3(x)
#         x = self.fc(self.flat(x))
        
#         return x

class AI85Net_DCT4(nn.Module):
    def __init__(self, num_channels=1, length=128, **kwargs):
        super().__init__()
    
        self.conv1bn = ai8x.FusedConv1dBNReLU(1, 16, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')

        self.conv2bn = ai8x.FusedConv1dBNReLU(16, 32, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')
        
        self.conv3bn = ai8x.FusedConv1dBNReLU(32, 64, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')

        self.conv4bn = ai8x.FusedConv1dBNReLU(64, 2, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')
                
        self.flat = nn.Flatten()
        self.fc = ai8x.Linear(128 * 2, 128)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1bn(x)
        x = self.conv2bn(x)
        x = self.conv3bn(x)
        x = self.conv4bn(x)
        x = self.fc(self.flat(x))
        # x = self.fc(x)
        
        return x
    
class AI85Net_DCT4_TORCHAUDIO(nn.Module):
    def __init__(self, bias=False, **kwargs):
        super().__init__()
    
        self.dct = ai8x.FusedConv1dAbs(128, 128, 1, stride=1, padding=0, bias=False, **kwargs)
        dct_coefs = create_dct(n_mfcc=128, n_mels=128, norm=None)
        with torch.no_grad():
            self.dct.op.weight = nn.Parameter(dct_coefs.transpose(0,1)[0:128,:].unsqueeze(2)/TORCHAUDIO_SQUEEZER, requires_grad=False)

    def forward(self, x):
        return self.dct(x)
    
def ai85net_dct4(pretrained=False, **kwargs):
    assert not pretrained
    return AI85Net_DCT4(**kwargs)

models = [
    {
        'name': 'ai85net_dct4',
        'min_input': 128,
        'dim': 1,
    },
]

