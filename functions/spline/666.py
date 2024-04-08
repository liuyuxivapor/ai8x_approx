import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

LEN = 2048
Q_LEN = 32768

x = np.linspace(0, 4 * np.pi, LEN)
y = np.sin(x)

cs = CubicSpline(x, y)

xq = np.sort(np.random.uniform(min(x), max(x), Q_LEN))
yq = cs(xq)

# plt.scatter(x, y, label='Sine Data')
# plt.plot(xq, yq, label='Cubic Spline Interpolation', color='red')
# plt.title('Cubic Spline Interpolation of Sine Data')
# plt.legend()
# plt.show()
