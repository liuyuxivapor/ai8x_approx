import torch
import numpy as np
import torch.nn as nn
from fft_mlp import *

is_test = True
is_gpu = torch.cuda.is_available()
    
X = torch.tensor(np.load("data_train.npy"), dtype=torch.float32)
y = torch.tensor(np.load("label_train.npy"), dtype=torch.float32)
model = mlp(len=64)

# model training    
if is_gpu:
    model.to(device='cuda')
    X = X.to(device='cuda')
    y = y.to(device='cuda')
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 250
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# test cases
if is_test:
    X_test = torch.tensor(np.load("data_test.npy"), dtype=torch.float32)
    y_test = torch.tensor(np.load("label_test.npy"), dtype=torch.float32)
    
    if is_gpu:
        X_test = X_test.to(device='cuda')
        y_test = y_test.to(device='cuda')
        
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)

    # result analysis
    # miss_rate = torch.sum(torch.abs(test_outputs - y_test) > 0.5).item() / (64 * 64)
    # print(f"Miss Rate: {miss_rate * 100:.2f}%")

    mse_loss = criterion(test_outputs, y_test)
    print(f"MSE Loss: {mse_loss.item():.6f}")

    # average_relative_error = torch.mean(torch.abs((y_test - test_outputs) / y_test))
    # print(f"Average Relative Error: {average_relative_error.item() * 100:.2f}%")
