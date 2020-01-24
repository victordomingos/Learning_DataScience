import matplotlib.pyplot as plt
import numpy as np
from sklearn import (datasets,
                     metrics,
                     naive_bayes)
from sklearn import model_selection as skms

iris = datasets.load_iris()

is_versicolor = (iris.target == 1)  # Get a true/false array

tts_versicolor = skms.train_test_split(iris.data, is_versicolor, test_size=.33, random_state=21)

(vc_train_features, vc_test_features,
 vc_train_target, vc_test_target) = tts_versicolor

gnb = naive_bayes.GaussianNB()
fit = gnb.fit(vc_train_features, vc_train_target)
prob_true = fit.predict_proba(vc_test_features)[:, 1]

print(prob_true)

fpr, tpr, thresh = metrics.roc_curve(vc_test_target, prob_true)
auc = metrics.auc(fpr, tpr)
print(f"FPR: {fpr}")
print(f"TPR: {tpr}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fpr, tpr, "o--")
ax.set_title(f"1-Class Iris ROC curve\nAUC: {auc:.3f}")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")

investigate = np.array([1, 3, 5])

for i in investigate:
    th, f, t = thresh[i], fpr[i], tpr[i]
    ax.annotate(f'thresh = {th:.3f}',
                xy=(f + .01, t - .01),
                xytext=(f + .1, t),
                arrowprops={'arrowstyle': '->'})

plt.show()
