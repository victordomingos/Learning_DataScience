import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection as skms
from sklearn import (datasets,
                     metrics,
                     naive_bayes,
                     neighbors)

sep = "\n" + 79 * "-" + "\n"

diabetes = datasets.load_diabetes()
tts = skms.train_test_split(diabetes.data,
                            diabetes.target,
                            test_size=.25)

train_features, test_features, train_target, test_target = tts
print(sep)
print("Train features shape: ", train_features.shape)
print("Train target shape: ", train_target.shape)
print("\nTest features shape: ", test_features.shape)
print("Test target shape: ", test_target.shape)
print(sep)

# features: BMI (body mass index, bp (blood pressure) s1-s6 (blood serum measurements)
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# target: a numerical score measuring the progression of a patient illness
diabetes_df['target'] = diabetes.target

print(diabetes_df.head())

sns.set(style="ticks", color_codes=True)
sns.pairplot(diabetes_df[['age', 'sex', 'bmi', 'bp', 's1']],
             height=1.5,
             # hue='sex',
             vars=diabetes_df.columns[:5],
             plot_kws={'alpha':.2})

plt.show()


knn = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = knn.fit(train_features, train_target
              )
predictions = fit.predict(test_features)
print(sep+"preds:", predictions.shape)

print(sep+"MSE", metrics.mean_squared_error(test_target, predictions))
print("TARGET VALUES INTERVAL DIMENSION:", diabetes_df['target'].max()-diabetes_df['target'].min())
print("SQRT(MSE)", np.sqrt(metrics.mean_squared_error(test_target, predictions)))
