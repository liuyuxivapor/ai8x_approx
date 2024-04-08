import numpy as np

# # 从二进制文件加载数据
# with open('c_array.bin', 'rb') as file:
#     c_array = np.fromfile(file, dtype=np.float32, count=2048)

# # 保存数据为.npy文件
# np.save('data.npy', c_array)

import numpy as np
import torch

# Specify the file names
input_file_name = "train.y"
output_file_name = "label_train.npy"

# Read data from the input file
data = np.loadtxt(input_file_name)

# Convert NumPy array to PyTorch tensor and reshape
# tensor_data = torch.tensor(data, dtype=torch.float32).view(-1, 1)
tensor_data = torch.tensor(data, dtype=torch.float32)
print(tensor_data.shape)

# Save the data as a NumPy array in the output file
np.save(output_file_name, tensor_data)
