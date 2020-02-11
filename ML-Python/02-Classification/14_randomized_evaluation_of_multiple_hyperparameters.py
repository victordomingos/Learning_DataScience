import numpy as np
import scipy.stats as ss
from sklearn import (datasets, model_selection as skms,
                     neighbors)

iris = datasets.load_iris()

knn = neighbors.KNeighborsClassifier()
param_dists = {'n_neighbors': np.arange(1, 11),
               'weights': ['uniform', 'distance'],
               'p': ss.geom(p=.5)}

model = skms.RandomizedSearchCV(knn,
                                param_distributions=param_dists,
                                cv=10,
                                n_iter=20,  # Nr. of times to sample
                                n_jobs=8)

model.fit(iris.data, iris.target)
print("\n\nBest Estimator:", model.best_estimator_,
      "\n\nBest Score:", model.best_score_,
      "\n\nBest Params:", model.best_params_)
