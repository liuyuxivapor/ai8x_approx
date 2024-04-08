import torch
import numpy as np
import math

data = torch.tensor(np.load('data_train.npy'))[8]
label = torch.tensor(np.load('label_train.npy'))[8]

size = data.size(1)

data_array = data[0].numpy().tolist()
label_array = label[0].numpy().tolist()

# 生成 C 语言代码字符串
c_code = f"""
#include <stdio.h>

float data_array[{size}] = {{ {', '.join(map(str, data_array))} }};
float label_array[{size}] = {{ {', '.join(map(str, label_array))} }};

int main() {{
    // 打印 C 数组
    for (int i = 0; i < {size}; ++i) {{
        printf("%f ", data_array[i]);
    }}

    return 0;
}}
"""

# 将 C 代码写入文件
with open('output.c', 'w') as file:
    file.write(c_code)
