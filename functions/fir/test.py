import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# import d2l
# from d2l import torch as d2l
from mlp import *

is_test = False
is_gpu = torch.cuda.is_available()

inputs = torch.tensor(np.load("input.npy"), dtype=torch.float32)
targets = torch.tensor(np.load("ref_output.npy"), dtype=torch.float32)
train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = fir_mlp(32)

if is_gpu:
    model.to(device='cuda')
    inputs = inputs.to(device='cuda')
    targets = targets.to(device='cuda')
    
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    for X, y in train_loader:
        if is_gpu:
            X = X.to(device='cuda')
            y = y.to(device='cuda')
            
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')
        

# d2l.plot(list(range(1, num_epochs+1)), losses, 'epoch', 'loss')
# d2l.plt.show()
