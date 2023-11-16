import numpy as np

# # 从二进制文件加载数据
# with open('c_array.bin', 'rb') as file:
#     c_array = np.fromfile(file, dtype=np.float32, count=2048)

# # 保存数据为.npy文件
# np.save('data.npy', c_array)

import torch

batch_size = 128
test_size = 64
len = 64

data_train = torch.randn(batch_size, 1, len, dtype=torch.float32)
data_test = torch.randn(test_size, 1, len, dtype=torch.float32)

def gen_data(inputs, batch, length):
    tmp1 = torch.zeros(batch, length, dtype=torch.float32)
    tmp2 = torch.zeros(batch, length, dtype=torch.float32)
    outputs = torch.zeros(batch, 1, 2 * length, dtype=torch.float32)
    
    for i in range(batch):
        fft_result = torch.fft.fftn(inputs[i])
        tmp1[i] = torch.real(fft_result)
        tmp2[i] = torch.imag(fft_result)
        outputs[i, 0, :] = torch.cat((tmp1[i], tmp2[i]), dim=0)
        
    return outputs

label_train = gen_data(data_train, batch_size, len)
label_test = gen_data(data_test, test_size, len)
    
np.save('data_train.npy', data_train)
np.save('label_train.npy', label_train)
np.save('data_test.npy', data_test)
np.save('label_test.npy', label_test)