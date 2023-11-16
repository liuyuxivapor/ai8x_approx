import numpy as np
import torch
import torch.nn as nn
from torchaudio__ import *

inputs = torch.tensor(np.load("data_train.npy"), dtype=torch.float32)
targets = torch.tensor(np.load("label_train.npy"), dtype=torch.float32)

inputs = inputs.reshape(32, 128, 1)
targets = targets.reshape(32, 128, 1)
print(inputs.shape, targets.shape)

model = DCT4_TORCHAUDIO()
    
criterion = nn.MSELoss()
       
outputs = model(inputs)
print(outputs.shape)
loss = criterion(outputs, targets)

print(f'Loss: {loss.item():}')

