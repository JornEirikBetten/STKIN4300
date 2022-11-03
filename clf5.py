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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree

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
filename = "PimaIndiansDiabetes2.csv" # updated dataset

df = pd.read_csv(datapath+filename)
df = df.drop(["Unnamed: 0"], axis=1)
df= df.dropna() # remove index column
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

"""
k-Nearest Neighbors (only k=16)
"""
k = 16
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
preds_train = knn.predict(x_train)
preds = knn.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print("16-Nearest Neighbours:")
print(f"AccTrain={AccTrain}, AccTest={AccTest}")

"""
Generalized Additive Model
"""
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

"""
fig = plt.figure(figsize=(8,5))
plt.plot([i for i in range(1, nfeatures+1, 1)], AIC_scores, label="AIC")
plt.xlabel("Number of features")
plt.ylabel("AIC score")
plt.vlines(x=np.argmin(AIC_scores)+1, ymin=min(AIC_scores), ymax=560, linestyle="dashed", label="Minimum AIC")
plt.legend()
#plt.savefig(fig_path("forward_selectionGAMclassification.pdf"), format="pdf", bbox_inches="tight")
plt.show()
"""

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
print(f"Features=['{best_features[0]}', '{best_features[1]}', '{best_features[2]}', '{best_features[3]}']")
print(f"Accuracy train: {AccTrain}")
print(f"Accuracy test: {AccTest}")

"""
Decision Tree Classifier
"""

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(x_train, y_train) # Define the cost complexity pruning path
ccp_alphas, impurities = path.ccp_alphas, path.impurities

K = 5
clfs = []
ETE = []
EPE = []
"""
5-fold cross validation for finding the best complexity parameter alpha
"""
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1337, ccp_alpha=ccp_alpha)
    cv_results = cross_validate(clf, x_train, y_train, cv=K, scoring="accuracy", return_train_score=True, return_estimator=True)
    estimators = cv_results["estimator"]
    test_scores = cv_results["test_score"]
    train_scores = cv_results["train_score"]
    EPE.append(np.mean(test_scores))
    ETE.append(np.mean(train_scores))
    clfs.append(estimators)



node_counts = [clf[0].tree_.node_count for clf in clfs]
depth = [clf[0].tree_.max_depth for clf in clfs]
"""
fig = plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("Number of nodes in tree")
#plt.savefig(fig_path("nodesvalphaclf.pdf"), format="pdf", bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
plt.xlabel(r'Cost complexity parameter $\alpha$')
plt.ylabel("Depth of tree")
#plt.savefig(fig_path("depthvalphaclf.pdf"), format="pdf", bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, ETE, color="tab:blue", marker="o", label="ETA", drawstyle="steps-post")
plt.plot(ccp_alphas, EPE, color="tab:orange", marker="o", label="EPA", drawstyle="steps-post")
plt.vlines(x=ccp_alphas[np.argmax(EPE)], ymin=0.6, ymax=np.max(EPE), color="tab:orange", linestyle="dashed", label="Maximum EPA")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("Accuracy score")
plt.legend()
#plt.savefig(fig_path("CVClfTree.pdf"), format="pdf", bbox_inches="tight")
plt.show()
"""
"""
Testing tree
"""
clf = DecisionTreeClassifier(random_state=1337, ccp_alpha=ccp_alphas[np.argmax(EPE)])
clf.fit(x_train, y_train)
preds_train = clf.predict(x_train)
preds = clf.predict(x_test)
AccTest = accuracy_score(y_test, preds)
AccTrain = accuracy_score(y_train, preds_train)
print("Decision tree classifier:")
print(f"Test: alpha={ccp_alphas[np.argmax(EPE)]}")
print(f"Train: ACC={AccTrain}")
print(f"Test: ACC={AccTest}")

"""
fig = plt.figure(figsize=(8, 5))
tree.plot_tree(clf, filled=True)
#plt.savefig(fig_path("ClfTree.pdf"), format="pdf", bbox_inches="tight")
plt.show()
"""


"""
Bagging
"""

clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, max_features=3, random_state=0)
clf.fit(x_train, y_train)
preds_train = clf.predict(x_train)
preds_probs = clf.predict_proba(x_test)
preds = clf.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print("Bagging probability: ")
print(f"Train: Acc={AccTrain}")
print(f"Test: Acc={AccTest}")

"""
Bagging no probability
"""
clf = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=10), n_estimators=100, max_features=3, random_state=0)
clf.fit(x_train, y_train)
preds_train = clf.predict(x_train)
preds_probs = clf.predict_proba(x_test)
preds = clf.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print("Bagging consensus: ")
print(f"Train: Acc={AccTrain}")
print(f"Test: Acc={AccTest}")

"""
Random Forest
"""

clf = RandomForestClassifier(max_depth=4, max_features="sqrt", random_state=0)
clf.fit(x_train, y_train)
preds_train = clf.predict(x_train)
preds = clf.predict(x_test)
AccTrain = accuracy_score(y_train, preds_train)
AccTest = accuracy_score(y_test, preds)

print("Random Forest: ")
print(f"Train: Acc={AccTrain}")
print(f"Test: Acc={AccTest}")


"""
AdaBoost
"""
lrs = np.logspace(-5, 1, 7)
for lr in lrs:
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), learning_rate=lr, random_state=0)
    clf.fit(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds = clf.predict(x_test)
    AccTrain = accuracy_score(y_train, preds_train)
    AccTest = accuracy_score(y_test, preds)

    print(f"AdaBoost: lr={lr}")
    print(f"Train: Acc={AccTrain}")
    print(f"Test: Acc={AccTest}")
