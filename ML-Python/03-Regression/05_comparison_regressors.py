import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import (model_selection as skms,
                     linear_model, preprocessing as skpre,
                     svm, tree, neighbors, metrics, pipeline)

student_df = pd.read_csv('student-mat.csv', sep=';').drop(columns="school")

features = student_df[student_df.columns[-10:-1]]
target = student_df[student_df.columns[-1:]]

# tts = skms.train_test_split(features, target)
# (train_features, test_features, train_target, test_target) = tts

old_school = [linear_model.LinearRegression(),
              neighbors.KNeighborsRegressor(n_neighbors=3),
              neighbors.KNeighborsRegressor(n_neighbors=10)]

# Penalized linear regression
plr = [linear_model.Lasso(), linear_model.Ridge()]

# Support Vector Regressors (defaults: epsilon=.1, nu=.5)
svrs = [svm.SVR(), svm.NuSVR()]

# Decision Trees
dtrees = [tree.DecisionTreeRegressor(max_depth=md)
          for md in [1, 3, 5, 10]]

# Merge all models above in a single list
models = old_school + plr + svrs + dtrees


def rms_error(actual, predicted):
    """ Root Mean Squared Error (less is better) """
    mse = metrics.mean_squared_error(actual, predicted)
    return np.sqrt(mse)


rms_scorer = metrics.make_scorer(rms_error)

scaler = skpre.StandardScaler()

scores = {}
for model in models:
    pipe = pipeline.make_pipeline(scaler, model)
    predictions = skms.cross_val_predict(pipe,
                                         features,
                                         target.values.ravel(),
                                         cv=10)

    model_name = str(model.__class__).split('.')[-1][:-2]
    key = (model_name +
           str(model.get_params().get('max_depth', '')) +
           str(model.get_params().get('n_neighbors', '')))
    scores[key] = rms_error(target, predictions)

df = pd.DataFrame.from_dict(scores, orient='index').sort_values(0)
df.columns = ['RMSE']
print(df)

print("----\n")

better_models = [tree.DecisionTreeRegressor(max_depth=3),
                 tree.DecisionTreeRegressor(max_depth=5),
                 linear_model.Ridge(),
                 linear_model.LinearRegression()]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for model in better_models:
    pipe = pipeline.make_pipeline(scaler, model)
    cv_results = skms.cross_val_score(pipe,
                                      features,
                                      target.values.ravel(),
                                      scoring=rms_scorer,
                                      cv=10)

    # str(model.__class__)... serves to isolate the model name
    model_name = str(model.__class__).split('.')[-1][:-2]
    model_name = (model_name
                  + str(model.get_params().get('max_depth', ''))
                  + str(model.get_params().get('n_neighbors', '')))
    label = f"{model_name.ljust(23)} {cv_results.mean():5.3f} ±{cv_results.std():.2f}"
    print(label)
    ax.plot(cv_results, 'o--', label=label)
    ax.set_xlabel("CrossValidation Fold #")
    ax.set_ylabel("RMSE")
    ax.legend()

plt.show()
