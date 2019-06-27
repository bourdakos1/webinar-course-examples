from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes(projection='3d')

_w1 = np.linspace(0, 3.6, 1000)
_b1 = np.linspace(0, 64, 1000)
w1, b1 = np.meshgrid(_w1, _b1)

x1 = -10.0
target1 = 14.0

x2 = 15.0
target2 = 59.0

x3 = 30.0
target3 = 86.0

y1 = x1 * w1 + b1
cost1 = (target1 - y1) ** 2

y2 = x2 * w1 + b1
cost2 = (target2 - y2) ** 2

y3 = x3 * w1 + b1
cost3 = (target3 - y3) ** 2

ax.scatter(1.8, 32, c='red')

# ax.plot_surface(w1, b1, cost1, cmap="plasma")
ax.plot_surface(w1, b1, (cost1 + cost2 + cost3) / 3, cmap="plasma")

plt.show()