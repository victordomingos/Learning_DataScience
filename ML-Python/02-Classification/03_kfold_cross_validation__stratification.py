from sklearn import (model_selection as skms,
                     datasets,
                     neighbors)

iris = datasets.load_iris()

model = neighbors.KNeighborsClassifier(10)

# 5-fold CV:
score = skms.cross_val_score(model,
                             iris.data,
                             iris.target,
                             cv=5)

print(score.reshape(-1, 1))
