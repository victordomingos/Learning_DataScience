import numpy as np
import pandas as pd

n_people = np.array([2, 3])
n_beers = np.array([0, 1, 2])
n_hotdogs = np.array([2, 4])

costs = np.array([80, 10, 5])

# numpy kung fu from the book (cartesian product):
counts = np.stack(np.meshgrid(n_people, n_beers, n_hotdogs), axis=-1).reshape(-1, 3)

totals = np.dot(counts, costs) + 40

cols = ["People", "Beer", "Hotdogs", "TotalCost"]

print(pd.DataFrame(np.column_stack([counts, totals]), columns=cols).head(8))
