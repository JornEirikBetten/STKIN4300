import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split, LeaveOneOut
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
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
from pygam import LinearGAM, PoissonGAM, s, te, LogisticGAM

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
order = 3 # order of basis functions
nsplines = features.shape[1]+order+1 # number of basis functions

best_features = []
best_features_idx = []
x_tr = x_train.to_numpy(); x_te = x_test.to_numpy()
nfeatures = x_train.shape[1]
AIC_scores = []
AICs = np.zeros(nfeatures)
for i in range(nfeatures):
    gam = LogisticGAM(s(i, n_splines=nsplines, spline_order=order))
    gam.fit(x_train, y_train)
    AICs[i] = gam.statistics_["AIC"]
first_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[first_feature])
best_features_idx.append(first_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
second_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[second_feature])
best_features_idx.append(second_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature or i==second_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(second_feature, n_splines=nsplines, spline_order=order)+ \
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
third_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[third_feature])
best_features_idx.append(third_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature or i==second_feature:
        continue
    elif i==third_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(second_feature, n_splines=nsplines, spline_order=order)+ \
                          s(third_feature, n_splines=nsplines, spline_order=order)+ \
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
fourth_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[fourth_feature])
best_features_idx.append(fourth_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature or i==second_feature:
        continue
    elif i==third_feature or i==fourth_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(second_feature, n_splines=nsplines, spline_order=order)+ \
                          s(third_feature, n_splines=nsplines, spline_order=order)+ \
                          s(fourth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
fifth_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[fifth_feature])
best_features_idx.append(fifth_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature or i==second_feature:
        continue
    elif i==third_feature or i==fourth_feature:
        continue
    elif i==fifth_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(second_feature, n_splines=nsplines, spline_order=order)+ \
                          s(third_feature, n_splines=nsplines, spline_order=order)+ \
                          s(fourth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(fifth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
sixth_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[sixth_feature])
best_features_idx.append(sixth_feature)
print(best_features)

AICs = np.ones(nfeatures)*100000
for i in range(nfeatures):
    if i==first_feature or i==second_feature:
        continue
    elif i==third_feature or i==fourth_feature:
        continue
    elif i==fifth_feature or i==sixth_feature:
        continue
    else:
        gam = LogisticGAM(s(first_feature, n_splines=nsplines, spline_order=order)+\
                          s(second_feature, n_splines=nsplines, spline_order=order)+ \
                          s(third_feature, n_splines=nsplines, spline_order=order)+ \
                          s(fourth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(fifth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(sixth_feature, n_splines=nsplines, spline_order=order)+ \
                          s(i, n_splines=nsplines, spline_order=order))
        gam.fit(x_train, y_train)
        AICs[i] = gam.statistics_["AIC"]
seventh_feature = np.argmin(AICs)
AIC_scores.append(min(AICs))
best_features.append(feature_names[seventh_feature])
best_features_idx.append(seventh_feature)
print(best_features)

gam = LogisticGAM(s(0, n_splines=nsplines, spline_order=order)+s(1, n_splines=nsplines, spline_order=order)+\
                  s(2, n_splines=nsplines, spline_order=order)+s(3, n_splines=nsplines, spline_order=order)+\
                  s(4, n_splines=nsplines, spline_order=order)+s(5, n_splines=nsplines, spline_order=order)+\
                  s(6, n_splines=nsplines, spline_order=order)+s(7, n_splines=nsplines, spline_order=order))

gam.fit(x_train, y_train)
preds = gam.predict(x_test)
preds_train = gam.predict(x_train)
AccTest = accuracy_score(y_test, preds)
AccTrain = accuracy_score(y_train, preds_train)
print("Full model:")
print(f"Accuracy train: {AccTrain}")
print(f"Accuracy test: {AccTest}")
AIC_scores.append(gam.statistics_["AIC"]) # full model

fig = plt.figure(figsize=(8,5))
plt.plot([i for i in range(1, nfeatures+1, 1)], AIC_scores, label="AIC")
plt.xlabel("Number of features")
plt.ylabel("AIC score")
plt.vlines(x=np.argmin(AIC_scores)+1, ymin=min(AIC_scores), ymax=560, linestyle="dashed", label="Minimum AIC")
plt.legend()
plt.savefig(fig_path("forward_selectionGAMclassification.pdf"), format="pdf", bbox_inches="tight")


best_model = LogisticGAM(s(best_features_idx[0], n_splines=nsplines, spline_order=order)+\
                         s(best_features_idx[1], n_splines=nsplines, spline_order=order)+\
                         s(best_features_idx[2], n_splines=nsplines, spline_order=order)+\
                         s(best_features_idx[3], n_splines=nsplines, spline_order=order)+\
                         s(best_features_idx[4], n_splines=nsplines, spline_order=order))
best_model.fit(x_train, y_train)
preds = best_model.predict(x_test)
preds_train = best_model.predict(x_train)
AccTest = accuracy_score(y_test, preds)
AccTrain = accuracy_score(y_train, preds_train)

print("Best model found by stagewise forward selection:")
print(f"Features=['{best_features[0]}', '{best_features[1]}', '{best_features[2]}', '{best_features[3]}', '{best_features[4]}']")
print(f"Accuracy train: {AccTrain}")
print(f"Accuracy test: {AccTest}")
