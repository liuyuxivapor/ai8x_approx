import numpy as np
from scipy.interpolate import CubicSpline
# from d2l import torch as d2l
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 1, 4, 3, 5])

# Generate points for plotting the interpolated curve
x_interp = np.random(min(x), max(x), 5)
cs = CubicSpline(x, y)
y_interp = cs(y)

print(x_interp)

# Plot the original data and the cubic spline interpolation
plt.scatter(x, y, label='Data Points')
plt.plot(x_interp, y_interp, label='Cubic Spline Interpolation', color='red')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Cubic Spline Interpolation')
plt.show()
