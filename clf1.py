import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split, LeaveOneOut
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neighbors import DistanceMetric
from numpy.random import default_rng
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, accuracy_score
import sklearn.metrics
"""
plotting parameters
"""

fontsize = "large"
params = {"font.family": "serif",
          "font.sans-serif": ["Computer Modern"],
          "axes.labelsize": fontsize,
          "legend.fontsize": fontsize,
          "xtick.labelsize": fontsize,
          "ytick.labelsize": fontsize,
          "legend.handlelength": 2
          }

plt.rcParams.update(params)

"""
Loading and transforming data
"""

datapath = os.getcwd() + "/data/"
filename = "PimaIndiansDiabetes.csv"

df = pd.read_csv(datapath+filename)
df = df.drop(["Unnamed: 0"], axis=1) # remove index column
response = df["diabetes"]
features = df.drop(["diabetes"], axis=1)

feature_names = ["pregnant", "glucose", "pressure", "triceps", "insulin",\
                 "mass", "pedigree", "age"]
print(df)
response = np.where(response=="pos", 1, 0) # transform from "pos/neg" -> 1, 0

"""
Split in training and testing sets
"""
seed = 29292
x_train, x_test, y_train, y_test = train_test_split(features, response, test_size=0.33, random_state=seed)

print(x_train.shape)
print(x_test.shape)

"""
5-fold CV
"""

nvals = 40
CVscores_test = np.zeros(nvals-1)
CVscores_train = np.zeros(nvals-1)
AccTest = np.zeros(nvals-1)
K = 5
kvals = [k for k in range(1, nvals, 1)]
for k in kvals:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(knn, x_train, y_train, cv=K, return_train_score=True, return_estimator=True, scoring="accuracy", n_jobs=K)
    test_error = np.mean(cv_results["test_score"])
    train_error = np.mean(cv_results["train_score"])
    CVscores_test[k-1] = test_error
    CVscores_train[k-1] = train_error
    knn.fit(x_train, y_train)
    preds = knn.predict(x_test)
    AccTest[k-1] = accuracy_score(y_test, preds)



fig = plt.figure(figsize=(8,5))
plt.plot(kvals, CVscores_train, color="tab:blue", label="ETA")
plt.plot(kvals, CVscores_test, color="tab:orange", label="EPA")
plt.plot(kvals, AccTest, color="tab:red", label="Test accuracy")
plt.legend()
plt.xlabel("k")
plt.ylabel("Accuracy score")
plt.vlines(x=np.argmax(CVscores_test)+1, ymin=0.6, ymax=max(CVscores_test), color="tab:orange", label="Maximum predicted accuracy")
plt.xticks([i for i in range(1, nvals, 4)], [f"{i}" for i in range(1, nvals, 4)])
plt.savefig(fig_path("5CVknnDiabetes1.pdf"), format="pdf", bbox_inches="tight")
optimal_k = kvals[np.argmax(CVscores_test)]
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train)
preds_train = knn.predict(x_train)
preds = knn.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print(f"5-fold CV: k={optimal_k}")
print(f"AccTest={AccTest}, AccTrain={AccTrain}")

"""
Leave-one-out CV
"""
x_trainnp = x_train.to_numpy()
loo = LeaveOneOut()
loo.get_n_splits(x_train)
CVscores_test = np.zeros(nvals-1)
CVscores_train = np.zeros(nvals-1)
AccTest = np.zeros(nvals-1)
for k in kvals:
    knn = KNeighborsClassifier(n_neighbors=k)
    test_errors = []
    train_errors = []
    for train_index, test_index in loo.split(x_train):
        x_tr = x_trainnp[train_index]; x_te = x_trainnp[test_index]
        y_tr = y_train[train_index]; y_te = y_train[test_index]
        knn.fit(x_tr, y_tr)
        preds_train = knn.predict(x_tr)
        pred = knn.predict(x_te)
        train_errors.append(accuracy_score(y_tr, preds_train))
        test_errors.append(accuracy_score(y_te, pred))


    test_error = np.mean(test_errors)
    train_error = np.mean(train_errors)
    CVscores_test[k-1] = test_error
    CVscores_train[k-1] = train_error
    knn.fit(x_train, y_train)
    preds = knn.predict(x_test)
    AccTest[k-1] = accuracy_score(y_test, preds)

fig = plt.figure(figsize=(8,5))
plt.plot(kvals, CVscores_train, color="tab:blue", label="ETA")
plt.plot(kvals, CVscores_test, color="tab:orange", label="EPA")
plt.plot(kvals, AccTest, color="tab:red", label="Test accuracy")
plt.xlabel("k")
plt.ylabel("accuracy score")
plt.vlines(x=np.argmax(CVscores_test)+1, ymin=0.6, ymax=max(CVscores_test), color="tab:orange", label="Maximum predicted accuracy")
plt.legend()
plt.xticks([i for i in range(1, nvals, 4)], [f"{i}" for i in range(1, nvals, 4)])
plt.savefig(fig_path("LOOknnDiabates1.pdf"), format="pdf", bbox_inches="tight")

optimal_k = kvals[np.argmax(CVscores_test)]
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train)
preds_train = knn.predict(x_train)
preds = knn.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print(f"LOO CV: k={optimal_k}")
print(f"AccTest={AccTest}, AccTrain={AccTrain}")
