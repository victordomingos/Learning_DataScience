import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as skms
from sklearn import (datasets,
                     dummy,
                     metrics,
                     neighbors)

iris = datasets.load_iris()

tts = skms.train_test_split(iris.data, iris.target,
                            test_size=.33,
                            random_state=21)

train_features, test_features, train_target, test_target = tts

strategies = ['constant', 'uniform', 'stratified', 'prior', 'most_frequent']
args = [{'strategy': s} for s in strategies]
args[0]['constant'] = 0  # class 0 is setosa

accuracies = []
for arg in args:
    baseline = dummy.DummyClassifier(**arg)
    baseline.fit(train_features, train_target)
    predictions = baseline.predict(test_features)
    accuracies.append(metrics.accuracy_score(predictions, test_target))
    print(predictions)
    print(arg, "Accuracy:", accuracies[-1])


print("\n", pd.DataFrame({'accuracy': accuracies}, index=strategies))