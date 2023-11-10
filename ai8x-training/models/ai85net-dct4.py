import torch
import torch.nn as nn

import ai8x

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
    def __init__(self, num_channels=1, length=512, **kwargs):
        super().__init__()
    
        self.conv1bn = ai8x.FusedConv1dBNReLU(1, 8, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')

        self.conv2bn = ai8x.FusedConv1dBNReLU(8, 16, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')
        
        self.conv3bn = ai8x.FusedConv1dBNReLU(16, 2, kernel_size=3, stride=1, padding=1, bias=True, batchnorm='Affine')
        
        self.flat = nn.Flatten()
        self.fc = ai8x.Linear(512 * 2, 512)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1bn(x)
        x = self.conv2bn(x)
        x = self.conv3bn(x)
        x = self.fc(self.flat(x))
        # x = self.fc(x)
        
        return x
    
def ai85net_dct4(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_DCT4(**kwargs)

models = [
    {
        'name': 'ai85net_dct4',
        'min_input': 512,
        'dim': 1,
    },
]
