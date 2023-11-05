import numpy as np
import torch
from d2l import torch as d2l

# Load the data from 'epoch_loss_data.npy'
loss_data = torch.tensor(np.load('./dct4/loss_data.npy'), dtype=torch.float32)

epoch = torch.arange(1000)

# Plot the data using d2l
d2l.plot(epoch, loss_data, 'epoch', 'loss')
d2l.plt.show()
