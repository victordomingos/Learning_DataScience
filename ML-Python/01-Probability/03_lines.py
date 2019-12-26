import matplotlib.pyplot as plt
import numpy as np

people = np.arange(1, 11)
total_cost = np.ones_like(people) * 40

ax = plt.gca()
ax.plot(people, total_cost)

ax.set_xlabel("Number of  People")
ax.set_ylabel("Cost")

plt.xlim(0, 11)
plt.show()
