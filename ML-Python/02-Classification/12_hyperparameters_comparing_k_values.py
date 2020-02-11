import matplotlib.pyplot as plt
import pandas as pd
from sklearn import (datasets, model_selection as skms,
                     neighbors)

digits = datasets.load_digits()

# Let's compare different values of k
param_grid = {'n_neighbors': [1, 3, 5, 10, 20]}
knn = neighbors.KNeighborsClassifier()
grid_model = skms.GridSearchCV(knn,
                               return_train_score=True,
                               param_grid=param_grid,
                               cv=10,
                               n_jobs=8)
# (using accuracy)

grid_model.fit(digits.data, digits.target)
# print(grid_model.cv_results_)

# using pandas to get a better overview over the results

param_cols = ['param_n_neighbors']
score_cols = ['mean_train_score', 'std_train_score',
              'mean_test_score', 'std_test_score']

pd.options.display.width = 0  # ignore terminal width (expand all columns of dataframe)
df = pd.DataFrame(grid_model.cv_results_).head()
print(df[param_cols + score_cols])

# Extract just the interesting columns
grid_df = pd.DataFrame(grid_model.cv_results_,
                       columns=['param_n_neighbors',
                                'mean_train_score',
                                'mean_test_score'])
grid_df.set_index('param_n_neighbors', inplace=True)
print(grid_df)

# Show a nice graph comparing accuracy across those five k values
ax = grid_df.plot.line(marker='.')
ax.set_xticks(grid_df.index)

plt.show()
