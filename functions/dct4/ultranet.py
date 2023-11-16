import torch
import torch.nn as nn

class ultranet(nn.Module):
    def __init__(self):
        super(ultranet, self).__init__()
        
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

#             nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

#             nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),

            nn.Conv1d(64, 2, kernel_size=1, stride=1, padding=0, bias=False),
            
            nn.Flatten(),
            nn.Linear(128 * 2, 128)
            
        )

    def forward(self, x):
        x = self.layers(x)
        return x

