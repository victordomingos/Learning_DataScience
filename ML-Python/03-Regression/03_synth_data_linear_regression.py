import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import (model_selection as skms,
                     datasets,
                     metrics,
                     linear_model)

#np.random.seed(42)
N = 20

features = np.linspace(-10, 10, num=N)
targets = 2 * features ** 2 - 3 + np.random.uniform(-2, 2, N)  # targets = func(features)

tts = skms.train_test_split(features, targets, test_size=N // 2)

train_features, test_features, train_target, test_target = tts

print(pd.DataFrame({"features": features, "targets": targets}))

plt.plot(train_features, train_target, 'bo')
plt.plot(test_features, np.zeros_like(test_features), 'r+')
plt.show()

lr = linear_model.LinearRegression()
lr.fit(train_features.reshape(-1,1), train_target)
predictions = lr.predict(test_features.reshape(-1,1))
print(predictions[:3])