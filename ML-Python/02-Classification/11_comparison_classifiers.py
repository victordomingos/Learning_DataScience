import itertools

import matplotlib.pyplot as plt
from sklearn import (datasets, model_selection as skms,
                     naive_bayes, linear_model, discriminant_analysis, svm, tree, neighbors, dummy)

digits = datasets.load_digits()

print("Shape:", digits.images[0].shape)
print("Target value of example #0: ", digits.target[0])

# plt.figure(figsize=(3, 3))
# plt.imshow(digits.images[0], cmap='gray')
# plt.show()  # This is supposed to be a zero

# ==============

classifiers = {  # 'LogReg(1)': linear_model.LogisticRegression(max_iter=1000),
    'LogReg(2)': linear_model.SGDClassifier(max_iter=1000, loss='log'),

    # 'QDA': discriminant_analysis.QuadraticDiscriminantAnalysis(),
    'LDA': discriminant_analysis.LinearDiscriminantAnalysis(),
    'GNB': naive_bayes.GaussianNB(),

    'SVC(1)': svm.SVC(kernel='linear'),
    # 'SVC(2)': svm.LinearSVC(),

    'DTC': tree.DecisionTreeClassifier(),
    '5NN-C': neighbors.KNeighborsClassifier(),
    '10NN-C': neighbors.KNeighborsClassifier(n_neighbors=10)
}

baseline = dummy.DummyClassifier(strategy='uniform')

base_score = skms.cross_val_score(baseline,
                                  digits.data,
                                  digits.target == 1,
                                  cv=10,
                                  scoring='average_precision',
                                  n_jobs=-1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(base_score, label='base')

markers = itertools.cycle(['+', '^', 'o', '_', '*', 'd', 'x', 's'])

for name, model in classifiers.items():
    cv_scores = skms.cross_val_score(model,
                                     digits.data, digits.target,
                                     cv=10,
                                     scoring='f1_macro',
                                     n_jobs=-1)

    label = f'{name} {cv_scores.mean():.3f}'
    ax.plot(cv_scores, label=label, marker=next(markers))

ax.set_ylim(0.0, 1.1)
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy')
ax.legend(loc='lower center', ncol=2)
plt.show()
