import numpy as np
import pandas as pd
from sklearn import (datasets, model_selection as skms,
                     neighbors)

digits = datasets.load_digits()
iris = datasets.load_iris()

param_grid = {'n_neighbors': np.arange(1, 10),
              'weights': ['uniform', 'distance'],
              'p': [1, 2, 4, 8, 16]}

knn = neighbors.KNeighborsClassifier()

grid_model = skms.GridSearchCV(knn, param_grid=param_grid, cv=10, n_jobs=8)

# this would take some time...
# grid_model.fit(digits.data, digits.target)
grid_model.fit(iris.data, iris.target)

param_df = pd.DataFrame.from_records(grid_model.cv_results_['params'])
param_df['mean_test_score'] = grid_model.cv_results_['mean_test_score']

print("\n\nSo, which sets of hyperparameters performed the best?\n")
print(param_df.sort_values(by=['mean_test_score']).tail())

print("\n\nBest Estimator:", grid_model.best_estimator_,
      "\n\nBest Score:", grid_model.best_score_,
      "\n\nBest Params:", grid_model.best_params_)
