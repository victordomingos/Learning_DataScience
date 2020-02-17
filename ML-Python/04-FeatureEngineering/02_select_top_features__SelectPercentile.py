import numpy as np
import pandas as pd
from sklearn import (datasets,
                     feature_selection as ftr_sel,
                     model_selection as skms)

diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

tts = skms.train_test_split(diabetes.data, diabetes.target,
                            test_size=.25,
                            random_state=42)

train_features, test_features, train_target, test_target = tts

# feature_select = ftr_sel.SelectPercentile(ftr_sel.mutual_info_regression, percentile=25)
feature_select = ftr_sel.SelectPercentile(ftr_sel.f_regression, percentile=25)

feature_select.fit_transform(train_features, train_target)

keepers_index = feature_select.get_support()
print(np.array(diabetes.feature_names)[keepers_index])
