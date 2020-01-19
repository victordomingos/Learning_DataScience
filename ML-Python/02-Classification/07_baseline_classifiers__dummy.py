from sklearn import (datasets,
                     dummy,
                     metrics)
from sklearn import model_selection as skms

iris = datasets.load_iris()

tts = skms.train_test_split(iris.data, iris.target,
                            test_size=.33,
                            random_state=21)

train_features, test_features, train_target, test_target = tts

baseline = dummy.DummyClassifier(strategy="most_frequent")
baseline.fit(train_features, train_target)
predictions = baseline.predict(test_features)
accuracy = metrics.accuracy_score(predictions, test_target)
print(predictions)
print("(Dummy/MostFrequent) Accuracy:", accuracy)
