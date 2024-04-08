import numpy as np
from scipy.fftpack import dct
import torch

seed_value = 36
np.random.seed(seed_value)
SCALER = 0.05

x_train = np.random.rand(32768) * SCALER
y_train = dct(x_train, type=4, norm='ortho')

x_test = np.random.rand(2048) * SCALER
y_test = dct(x_test, type=4, norm='ortho')

x_train_data = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_data = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_test_data = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)
y_test_data = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# print("Min value of x_train_normalized:", np.min(y_train_data_scaled))
# print("Max value of x_train_normalized:", np.max(y_train_data_scaled))

# print("Min value of x_test_normalized:", np.min(y_test_data_scaled))
# print("Max value of x_test_normalized:", np.max(y_test_data_scaled))


np.save('data_train.npy', x_train_data)
np.save('label_train.npy', y_train_data)
np.save('data_test.npy', x_test_data)
np.save('label_test.npy', y_test_data)