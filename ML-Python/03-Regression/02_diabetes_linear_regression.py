import numpy as np
import pandas as pd
from sklearn import (model_selection as skms,
                     datasets,
                     metrics,
                     linear_model)

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


lr = linear_model.LinearRegression()
fit = lr.fit(train_features, train_target)
predictions = fit.predict(test_features)

print(sep + "preds:", predictions.shape)

print(sep + "MSE", metrics.mean_squared_error(test_target, predictions))
print("TARGET VALUES INTERVAL DIMENSION:", diabetes_df['target'].max() - diabetes_df['target'].min())
print("SQRT(MSE)", np.sqrt(metrics.mean_squared_error(test_target, predictions)))
