import matplotlib.pyplot as plt
from sklearn import model_selection as skms
from sklearn import (datasets,
                     naive_bayes,
                     neighbors)

iris = datasets.load_iris()

classificadores = {'gnb': naive_bayes.GaussianNB(),
                   '5-NN': neighbors.KNeighborsClassifier(n_neighbors=5)}

fig, ax = plt.subplots(figsize=(6, 4))
for nome, modelo in classificadores.items():
    cv_scores = skms.cross_val_score(modelo,
                                     iris.data, iris.target,
                                     cv=10,
                                     scoring='accuracy',
                                     n_jobs=-1)  # utilizar todos os núcleos

    lbl = "{} {:.3f}".format(nome, cv_scores.mean())
    ax.plot(cv_scores, "-o", label=lbl)

ax.set_ylim(0.0, 1.1)
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.legend(ncol=2)
plt.show()
