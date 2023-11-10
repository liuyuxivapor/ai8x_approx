import torch
import torch.nn as nn
import d2l
from d2l import torch as d2l
from torchsummary import summary
from fft_mlp import *

is_test = False
is_gpu = torch.cuda.is_available()

def generate_data(num_samples, sig_length):
    # X = torch.randn(num_samples, 1, sig_length)
    # y = torch.zeros(num_samples, 2, sig_length)

    # for i in range(num_samples):
    #     fft_result = torch.fft.fftn(X[i, :, :])
    #     y[i, :, :] = torch.cat((fft_result.real, fft_result.imag), dim=0)

    # return X, y
    X = torch.randn(num_samples, 1)
    y = torch.zeros(num_samples, 2)

    for i in range(num_samples):
        fft_result = torch.fft.fftn(X[i, :])
        y[i, :] = torch.cat((fft_result.real, fft_result.imag), dim=0)

    return X, y
    
def init_weight(m):
    if type(m) == nn.Conv1d or type(m) == nn.BatchNorm1d:
        nn.init.uniform_(m.weight, -1.0, 1.0)
    
batch_size = 2048
X, y = generate_data(batch_size, sig_length=64)

num_epochs = 10000
model = mlp()
model.apply(init_weight)

# model training    
if is_gpu:
    model.to(device='cuda')
    X = X.to(device='cuda')
    y = y.to(device='cuda')
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

losses = []

for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        
d2l.plot(list(range(1, num_epochs+1)), losses, 'epoch', 'loss')
d2l.plt.show()
summary(model, (1,)) 

# test cases
if is_test:
    test_size = 32768
    
    X_test, y_test = generate_data(test_size)
    if is_gpu:
        X_test = X_test.to(device='cuda')
        y_test = y_test.to(device='cuda')
        
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)

    # result analysis
    miss_rate = torch.sum(torch.abs(test_outputs - y_test) > 0.5).item() / (test_size * 64)
    print(f"Miss Rate: {miss_rate * 100:.2f}%")

    mse_loss = criterion(test_outputs, y_test)
    print(f"MSE Loss: {mse_loss.item():.4f}")

    average_relative_error = torch.mean(torch.abs((y_test - test_outputs) / y_test))
    print(f"Average Relative Error: {average_relative_error.item() * 100:.2f}%")
