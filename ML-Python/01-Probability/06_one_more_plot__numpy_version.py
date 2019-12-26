import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(-3, 3, 100)  # 100 values from -3 to 3
xx_p1 = np.c_[xx, np.ones_like(xx)]  # Create an array with these two columns (2nd col filled with ones)

w = np.array([1.5, -3])

yy = np.dot(xx_p1, w)
# Calculate dot product of these two arrays. The same as:
# yy = xx_p1 @ w
# yy = xx_p1.dot(w)

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

plt.show()
