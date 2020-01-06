import pandas as pd

from sklearn import model_selection as skms
from sklearn import (datasets,
                     metrics,
                     naive_bayes,
                     neighbors)


sep = "\n" + 79*"-" + "\n"


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

print(sep, df)


# Get a smaller sample for training
tts = skms.train_test_split(iris.data,
                            iris.target,
                            test_size=.25)

train_features, test_features, train_target, test_target = tts

print(sep)
print("Train features shape: ", train_features.shape)
print("Train target shape: ", train_target.shape)
print("\nTest features shape: ", test_features.shape)
print("Test target shape: ", test_target.shape)
print(sep)


knn = neighbors.KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(train_features, train_target)
predictions = fit.predict(test_features)

print("Predictions:\n", predictions)
print('Test target:\nh', test_target)
print("3NN accuracy:", metrics.accuracy_score(test_target, predictions))
print(sep)


nb = naive_bayes.GaussianNB()
fit = nb.fit(train_features, train_target)
predictions = fit.predict(test_features)

print("Predictions:\n", predictions)
print('Test target:\n', test_target)
print("Naive-Bayes accuracy:", metrics.accuracy_score(test_target, predictions))

