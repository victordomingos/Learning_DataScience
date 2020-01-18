import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import (datasets,
                     neighbors)
from sklearn import model_selection as skms

iris = datasets.load_iris()

# 10 data set sizes (10% to 100%)
train_sizes = np.linspace(.1, 1.0, 10)

knc = neighbors.KNeighborsClassifier()

(train_n,
 train_scores,
 test_scores) = skms.learning_curve(knc,
                                    iris.data,
                                    iris.target,
                                    cv=5,
                                    train_sizes=train_sizes)

# 5 Cross-Validation scores (1 for each data set size)
df = pd.DataFrame(test_scores,
                  index=(train_sizes * 100).astype(np.int))
df['Média 5-CV'] = df.mean(axis='columns')
df.index.name = "% Dados Utilizados"
print("\n", df)

joined = np.array([train_scores, test_scores]).transpose()

ax = sns.tsplot(joined,
                time=train_sizes,
                condition=['Treino', 'Teste'],
                interpolate=False)

ax.set_title("Curva de aprendizagem para Classificador 5-NN")
ax.set_xlabel("Proporção de amostras utilizadas para treino")
ax.set_ylabel("Precisão")

plt.show()
