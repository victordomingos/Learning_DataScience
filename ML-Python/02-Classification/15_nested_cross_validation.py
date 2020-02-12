import numpy as np
from sklearn import (datasets, model_selection as skms,
                     neighbors)

iris = datasets.load_iris()

param_grid = {'n_neighbors': np.arange(1, 11),
              'weights': ['uniform', 'distance'],
              'p': [1, 2, 4, 8, 16]}

knn = neighbors.KNeighborsClassifier()

grid_knn = skms.GridSearchCV(knn,
                             param_grid=param_grid,
                             cv=2,
                             n_jobs=8)

outer_scores = skms.cross_val_score(grid_knn,
                                    iris.data, iris.target,
                                    cv=5)
print("Outer scores:", outer_scores)

# train our preferred model based on the params from GridSearchCV
grid_knn.fit(iris.data, iris.target)
preferred_params = grid_knn.best_estimator_.get_params()
print(preferred_params)
final_knn = neighbors.KNeighborsClassifier(**preferred_params)
final_knn.fit(iris.data, iris.target)
print(final_knn)

# now we could predict with this model…
