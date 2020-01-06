import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

pd.options.display.float_format = '{:20,.4f}'.format

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

print(pd.concat([df.head(3), df.tail(3)]))
print(df['species'].describe())

sns.pairplot(data=df, vars=df.columns[0:4], hue='species', height=1.5)
plt.show()
