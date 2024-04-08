import numpy as np
import torch

SCALER = 1

with open('x.bin', 'rb') as file1:
    x_array = np.fromfile(file1, dtype=np.float32, count=320)

x_data = torch.tensor(x_array, dtype=torch.float32).view(-1, 1) * SCALER
print(x_data.shape)
np.save('x.npy', x_data)


with open('y.bin', 'rb') as file2:
    y_array = np.fromfile(file2, dtype=np.float32, count=320) * SCALER

y_data = torch.tensor(y_array, dtype=torch.float32).view(-1, 1)
print(y_data.shape)
np.save('y.npy', y_data)
