import numpy as np

# 从二进制文件加载数据
with open('c_array.bin', 'rb') as file:
    c_array = np.fromfile(file, dtype=np.float32, count=2048)

# 保存数据为.npy文件
np.save('data.npy', c_array)
