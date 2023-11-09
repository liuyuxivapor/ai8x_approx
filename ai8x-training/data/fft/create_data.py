import numpy as np

# # 从二进制文件加载数据
# with open('c_array.bin', 'rb') as file:
#     c_array = np.fromfile(file, dtype=np.float32, count=2048)

# # 保存数据为.npy文件
# np.save('data.npy', c_array)

import torch

batch_size = 16
len = 2048

data_train = torch.randn(batch_size, len, dtype=torch.float32)
label_train = torch.zeros(batch_size, 2 * len, dtype=torch.float32)

tmp1 = torch.randn(batch_size, len, dtype=torch.float32)
tmp2 = torch.randn(batch_size, len, dtype=torch.float32)

for i in range(batch_size):
    fft_result = torch.fft.fftn(data_train[i, :])
    tmp1[i] = torch.real(fft_result)
    tmp2[i] = torch.imag(fft_result)
    label_train[i] = torch.cat(tmp1[i], tmp2[i], dim=0)
    

np.save('data_train.npy', data_train)
np.save('label_train.npy', label_train)

# data_test = torch.randn(batch_size, 1, len, dtype=torch.float32)
# label_test = torch.zeros(batch_size, 2, len, dtype=torch.float32)

# np.save('data_train.npy', data_train)
# np.save('label_train.npy', label_train)