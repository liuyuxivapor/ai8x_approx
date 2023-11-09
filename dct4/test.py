import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import d2l
from d2l import torch as d2l
from ultranet import *

is_test = False
is_gpu = torch.cuda.is_available()

inputs = torch.tensor(np.load("dct4_in.npy"), dtype=torch.float32)
batch_size = inputs.size(dim=0) # 16
len = inputs.size(dim=2)        # 512
targets = torch.tensor(np.load("dct4_out.npy"), dtype=torch.float32)
model = ultranet()

if is_gpu:
    model.to(device='cuda')
    inputs = inputs.to(device='cuda')
    targets = targets.to(device='cuda')
    summary(model, input_size=(1, len), device='cuda')
    
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []

num_epochs = 1000
for epoch in range(num_epochs):
    # for batch in range(batch_size):
    #     input = inputs[batch]
    #     target = targets[batch]
        
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():}')
        
# loss_data = np.array(losses)
# np.save('loss_data.npy'.format(epoch), loss_data)

d2l.plot(list(range(1, num_epochs+1)), losses, 'epoch', 'loss')
d2l.plt.show()
