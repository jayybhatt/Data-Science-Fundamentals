import time
import pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

pdb.set_trace()

train_df = pd.read_csv("./train_data_mod.csv", index_col=0)
train_y = train_df['logerror'].values
# cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
cat_cols = ["propertycountylandusecode", "propertyzoningdesc"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
feat_names = train_df.columns.values


X_train, X_valid, y_train, y_valid = train_test_split( train_df, train_y, test_size=0.3)

# #############################################################################
# Fit regression model
# train_size = 100
models = {}
models["svr"] = SVR(kernel='rbf', gamma=0.1)
cv=5
param_grid={}
param_grid["svr"] = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}

# for model in models:
svr = GridSearchCV(models["svr"], param_grid["svr"] )

# kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
# 				  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
# 							  "gamma": np.logspace(-2, 2, 5)})


t0 = time.time()
svr.fit(X_train, y_train)
svr_fit = time.time() - t0
print model
print("Complexity and bandwidth selected and model fitted in %.3f s"
	  % svr_fit)

# sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
# print("Support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_valid)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
	  % (X_valid.shape[0], svr_predict))

print mean_absolute_error(y_valid, svr_predict)

