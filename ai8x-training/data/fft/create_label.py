import numpy as np
import torch
import torch.fft
import os

# 读取.npy文件中的数据
data = np.load('/home/vapor/code/AIoT/ai8x-training/data/fft/data.npy')

# 将数据转换为PyTorch的Tensor
data_tensor = torch.tensor(data, dtype=torch.float32)

# 进行FFT计算
fft_result = torch.fft.fft(data_tensor)

# 提取FFT结果的实部和虚部
real_part = fft_result.real
imaginary_part = fft_result.imag

# 创建一个包含实部和虚部的新Tensor
fft_output = torch.cat((real_part, imaginary_part), dim=0)

# 保存新的Tensor为.npy文件
output_file = '/home/vapor/code/AIoT/ai8x-training/data/fft/fft_result.npy'
np.save(output_file, fft_output.cpu().numpy())

