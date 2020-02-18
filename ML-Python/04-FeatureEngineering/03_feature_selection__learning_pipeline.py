import pandas as pd
from sklearn import (datasets,
                     linear_model, pipeline, feature_selection as ftr_sel,
                     model_selection as skms
                     )

wine = datasets.load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

feature_select = ftr_sel.SelectPercentile(ftr_sel.mutual_info_classif,
                                          percentile=25)

pipe = pipeline.make_pipeline(feature_select, linear_model.SGDClassifier(n_jobs=8))

param_grid = {'selectpercentile__percentile': [5, 10, 15, 20, 25]}
grid = skms.GridSearchCV(pipe, param_grid=param_grid, cv=3)
grid.fit(wine.data, wine.target)

print(grid.best_params_)
print(grid.best_score_)
