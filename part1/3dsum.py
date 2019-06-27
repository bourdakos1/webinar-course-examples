from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes(projection='3d')

x = np.linspace(-20, 20, 1000)
y = np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(x, y)
Z = (X + Y)
ax.plot_surface(X, Y, Z, cmap="plasma")

plt.show()
