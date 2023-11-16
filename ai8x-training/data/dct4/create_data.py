import torch
import math
import numpy as np

batch = 64
test_size = 32
len = 128

def dct4(input_signal):
    
    batch = input_signal.size(dim=0)
    len = input_signal.size(dim=2)
    output_signal = torch.zeros(batch, len)

    for i in range(batch):
        for k in range(len):
            sum_value = 0.0
            for n in range(len):
                sum_value += input_signal[i, 0, n] * math.cos((math.pi / len) * (n + 0.5) * (k + 0.5))
            output_signal[i, k] = sum_value * 0.125

    return output_signal

data_train = torch.randn(batch, 1, len, dtype=torch.float32)
data_test = torch.randn(test_size, 1, len, dtype=torch.float32)
label_train = dct4(data_train)
label_test = dct4(data_test)

np.save('data_train.npy', data_train)
np.save('label_train.npy', label_train)
np.save('data_test.npy', data_test)
np.save('label_test.npy', label_test)

# import matplotlib.pyplot as plt

# # Assuming label_train is a torch tensor
# label_train_np = label_train.numpy()

# # Flatten the tensor to 1D array
# label_train_flattened = label_train_np.flatten()

# # Plotting the histogram
# plt.hist(label_train_flattened, bins='auto', alpha=0.7, color='blue', edgecolor='black')

# # Adding labels and title
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of label_train')

# # Show the plot
# plt.show()
