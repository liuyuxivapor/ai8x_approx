import torch
import torch.nn as nn

import ai8x

# class AI85Net_FPR(nn.Module):
#     """
#     5-Layer CNN that uses max parameters in AI84
#     """
#     def __init__(self, num_classes=10, num_channels=1, dimensions=(64, 64),
#                  planes=32, pool=2, fc_inputs=256, bias=False, **kwargs):
#         super().__init__()

#         # Limits
#         assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
#         assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

#         # Keep track of image dimensions so one constructor works for all image sizes
#         self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv2 = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=2, padding=1, bias=bias, **kwargs)
#         self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=2, padding=1, bias=bias, **kwargs)
#         self.conv4 = ai8x.FusedMaxPoolConv2d(64, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
#         # self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
#         #                                            stride=1, padding=1, bias=bias, **kwargs)

#         self.pooling=ai8x.MaxPool2d(8)
#         self.fc1 = ai8x.FusedLinearReLU(128, 128, bias=False, **kwargs)
#         self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         # print(x.size())
#         x = self.pooling(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc(x)

#         return x

class AI85Net_FPR(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=1, dimensions=(64, 64),
                 planes=32, pool=2, fc_inputs=256, bias=False, **kwargs):
        super().__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1, bias=False)
        # self.pooling1 = nn.MaxPool2d(2,2)
        self.bn4 = nn.BatchNorm2d(128)

        self.pooling = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(128, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        relu = nn.ReLU()
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        # x = self.pooling1(x)
        # print(x.size())
        x = self.pooling(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = relu(self.fc1(x))

        x = self.fc(x)

        return x

def ai85net_fpr(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_FPR(**kwargs)

models = [
    {
        'name': 'ai85net_fpr',
        'min_input': 1,
        'dim': 2,
    },

]
