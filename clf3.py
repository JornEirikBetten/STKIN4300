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

fig = plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("Number of nodes in tree")
plt.savefig(fig_path("nodesvalphaclf.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
plt.xlabel(r'Cost complexity parameter $\alpha$')
plt.ylabel("Depth of tree")
plt.savefig(fig_path("depthvalphaclf.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, ETE, color="tab:blue", marker="o", label="ETA", drawstyle="steps-post")
plt.plot(ccp_alphas, EPE, color="tab:orange", marker="o", label="EPA", drawstyle="steps-post")
plt.vlines(x=ccp_alphas[np.argmax(EPE)], ymin=0.6, ymax=np.max(EPE), color="tab:orange", linestyle="dashed", label="Maximum EPA")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("Accuracy score")
plt.legend()
plt.savefig(fig_path("CVClfTree.pdf"), format="pdf", bbox_inches="tight")

"""
Testing tree
"""
clf = DecisionTreeClassifier(random_state=1337, ccp_alpha=ccp_alphas[np.argmax(EPE)])
clf.fit(x_train, y_train)
preds_train = clf.predict(x_train)
preds = clf.predict(x_test)
AccTest = accuracy_score(y_test, preds)
AccTrain = accuracy_score(y_train, preds_train)
print(f"Test: alpha={ccp_alphas[np.argmax(EPE)]}")
print(f"Train: ACC={AccTrain}")
print(f"Test: ACC={AccTest}")

fig = plt.figure(figsize=(8, 5))
tree.plot_tree(clf, filled=True)
plt.savefig(fig_path("ClfTree.pdf"), format="pdf", bbox_inches="tight")



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
