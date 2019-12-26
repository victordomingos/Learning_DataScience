import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(-3, 3, 100)  # 100 values from -3 to 3
m = 1.5
b = -3

# y = mx + b
yy = m * xx + b

ax = plt.gca()

ax.plot(xx, yy)
ax.set_ylim(-4, 4)

# high-school look...
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal')

# bullets
ax.plot(0, -3, 'ro')  # y-intercept
ax.plot(2, 0, 'ro')

yy = 0 * xx + b
ax.plot(xx, yy, 'y')  # yellow line
plt.show()
