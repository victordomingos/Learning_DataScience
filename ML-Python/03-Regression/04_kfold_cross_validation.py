from sklearn import (model_selection as skms,
                     datasets,
                     neighbors)

diabetes = datasets.load_diabetes()

model = neighbors.KNeighborsRegressor(10)

# 5-fold CV:
score = skms.cross_val_score(model,
                             diabetes.data,
                             diabetes.target,
                             cv=5,
                             scoring='neg_mean_squared_error')

print(score.reshape(-1, 1))
