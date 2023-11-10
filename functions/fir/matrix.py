import torch
import torch.nn as nn
import math

# Omega(l) = 2 * pi * l * 6/48  (N = 28, l = 0 ... 28)
        
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.weight = nn.Parameter(torch.zeros((15, 29)))
        nn.init.uniform_(self.weight, -1.0, 1.0)
        
    def forward(self, x):
        k_range = 15
        l_range = 29
        cos_term = torch.zeros((l_range, k_range), dtype=torch.float32)
        a_term = torch.zeros(l_range, dtype=torch.float32)

        for l in range(l_range):
            for k in range(k_range):
                cos_term[l, k] = math.cos(k * 2 * math.pi * l * 6 / 48)
                
        cos_term = cos_term.to(device='cuda')
                                
        for l in range(l_range):
            a_term[l] = torch.matmul(cos_term[l, :], self.weight[:, l])
            
        a_term = a_term.to(device='cuda')

        return a_term
        