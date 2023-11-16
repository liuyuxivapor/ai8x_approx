import numpy as np
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

LOW = 1
HIGH = 128
LEN = 16
Q_LEN = 32

x = torch.randint(LOW, HIGH, size=(LEN,), dtype=torch.float32)
x = torch.sort(x).values
y = torch.randint(LOW, HIGH, size=(LEN,), dtype=torch.int32)
y = torch.sort(y).values
print(x, y)

xq = torch.randint(min(x), max(x), size=(Q_LEN,), dtype=torch.int32)
x1 = torch.sort(xq).values
cs = CubicSpline(x, y)
yq = cs(y)
# print(xq, yq)

plt.plot(x, y, 'o', label='Data points')
plt.plot(xq, yq, label='Cubic Spline Interpolation')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
