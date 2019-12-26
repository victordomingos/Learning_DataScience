import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

people = np.arange(1, 11)
total_cost = 80.0 * people + 40

# DataFrame.T --> translate rows as cols and vice versa
df = pd.DataFrame({'total_cost': total_cost.astype(np.int)}, index=people).T
# print(df.info())
# print(df.to_string())

ax = plt.gca()
ax.plot(people, total_cost, 'bo')
ax.set_xlabel("Number of People")
ax.set_ylabel("Cost")
plt.show()
