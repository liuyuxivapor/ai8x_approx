import numpy as np
import torch
import torch.nn as nn
import d2l
from d2l import torch as d2l
from matrix import *

is_test = False
is_gpu = torch.cuda.is_available()

inputs = torch.arange(29)
targets = torch.tensor(np.load("fir_lpf.npy"), dtype=torch.float32)
model = MyNetwork()

if is_gpu:
    model.to(device='cuda')
    inputs = inputs.to(device='cuda')
    targets = targets.to(device='cuda')
    
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():}')
        

d2l.plot(list(range(1, num_epochs+1)), losses, 'epoch', 'loss')
d2l.plt.show()
