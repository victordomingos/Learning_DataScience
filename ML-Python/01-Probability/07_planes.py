import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

n_people = np.arange(1, 11)
n_beers = np.arange(0, 20)

n_people, n_beers = np.meshgrid(n_people, n_beers)

total_cost = 80 * n_people + 10 * n_beers + 40

fig, axes = plt.subplots(2, 3,
                         subplot_kw={'projection': '3d'},
                         figsize=(9, 6))

angles = [0, 45, 90, 135, 180]

for ax, angle in zip(axes.flat, angles):
    ax.plot_surface(n_people, n_beers, total_cost)
    ax.set_xlabel("People")
    ax.set_ylabel("Beers")
    ax.set_zlabel("Total Cost")
    ax.azim = angle

plt.show()
