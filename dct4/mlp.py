import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self, len):
        super(MyNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(1 * len, 4 * len), 
            nn.ReLU(),
            nn.Linear(4 * len, 4 * len), 
            nn.ReLU(),
            nn.Linear(4 * len, 1 * len)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    

# class MyNetwork(nn.Module):
#     def __init__(self):
#         super(MyNetwork, self).__init__()
        
#         self.layers = nn.Sequential(
#             nn.Linear(1, 4), 
#             nn.ReLU(),
#             nn.Linear(4, 4), 
#             nn.ReLU(),
#             nn.Linear(4, 1)
#         )

#     def forward(self, x):
#         x = self.layers(x)
#         return x