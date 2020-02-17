import numpy as np
import pandas as pd
from sklearn import (datasets,
                     feature_selection as ftr_sel,
                     model_selection as skms)

wine = datasets.load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

tts = skms.train_test_split(wine.data, wine.target,
                            test_size=.25,
                            random_state=42)

train_features, test_features, train_target, test_target = tts

# feature_select = ftr_sel.SelectKBest(ftr_sel.mutual_info_classif, k=5)
feature_select = ftr_sel.SelectKBest(ftr_sel.f_classif, k=5)

feature_select.fit_transform(train_features, train_target)

keepers_index = feature_select.get_support()
print(np.array(wine.feature_names)[keepers_index])
