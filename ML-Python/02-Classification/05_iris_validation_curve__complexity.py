import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import (datasets,
                     neighbors)
from sklearn import model_selection as skms

iris = datasets.load_iris()

n_neighbors = [1, 3, 5, 10, 15, 20]
knc = neighbors.KNeighborsClassifier()

tt = skms.validation_curve(knc,
                           iris.data,
                           iris.target,
                           param_name='n_neighbors',
                           param_range=n_neighbors,
                           cv=5)
# stack and transpose
joined = np.array(tt).transpose()

ax = sns.tsplot(joined,
                time=n_neighbors,
                condition=['Treino', 'Teste'],
                interpolate=False)

ax.set_title("Desempenho CV 5-fold para k-NN")
ax.set_xlabel("k para k-NN:\n(quanto maior o valor de k, mais complexo")
ax.set_ylim(.9, 1.01)
ax.set_ylabel("Precisão")

plt.show()
