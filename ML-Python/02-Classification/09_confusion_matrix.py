import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import (datasets,
                     metrics,
                     neighbors)
from sklearn import model_selection as skms

iris = datasets.load_iris()

tts = skms.train_test_split(iris.data, iris.target,
                            test_size=.33,
                            random_state=21)

train_features, test_features, train_target, test_target = tts

target_predictions = (neighbors.KNeighborsClassifier()
                      .fit(train_features, train_target)
                      .predict(test_features))

cm = metrics.confusion_matrix(test_target, target_predictions)

print("\nConfusion Matrix:", cm, sep="\n\n")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax = sns.heatmap(cm, annot=True, square=True,
                 xticklabels=iris.target_names,
                 yticklabels=iris.target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plt.show()
