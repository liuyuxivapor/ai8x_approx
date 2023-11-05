import torch
import math
import numpy as np

batch_size = 16
len = 512

def dct4(input_signal):
    
    batch_size = input_signal.size(dim=0)
    len = input_signal.size(dim=1)
    output_signal = torch.zeros(batch_size, len)

    for i in range(batch_size):
        for k in range(len):
            sum_value = 0.0
            for n in range(len):
                sum_value += input_signal[i, n] * math.cos((math.pi / len) * (n + 0.5) * (k + 0.5))
            output_signal[i, k] = sum_value * 0.0625

    return output_signal

input_signal = torch.randn(batch_size, len, dtype=torch.float32)
# print(torch.mean(input_signal),
#       torch.var(input_signal))
dct_result = dct4(input_signal)
# print(dct_result.shape)
np.save('dct4_in.npy', input_signal)
np.save('dct4_out.npy', dct_result)

