import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import sklearn.metrics
from sklearn.tree import DecisionTreeRegressor
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
Data
"""

datapath = os.getcwd() + "/data/"
labels_wI = ["Intercept", "TPSA", "SAacc", "H050", "MLOGP", "RDCHI", \
                    "GATS1p", "nN", "C040"]
dependent_labels = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", \
                    "GATS1p", "nN", "C040"]
filename = "qsar_aquatic_toxicity.csv"
dichotomize_labels=["H-050", "nN", "C-040"]
qsar = QSAR(datapath + filename, dichotomize_labels)
seed = 1337
rng = default_rng(seed)

x = qsar.x # non-dichotomized
xD = qsar.xdich # dichotomized
y = qsar.y # response

"""
Split in train and test
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
reg = DecisionTreeRegressor(random_state=0)
path = reg.cost_complexity_pruning_path(x_train, y_train) # Define the cost complexity pruning path
ccp_alphas, impurities = path.ccp_alphas, path.impurities
K = 5
regs = []
ETE = []
EPE = []
"""
Cross validation for finding the best complexity parameter alpha
"""
for ccp_alpha in ccp_alphas:
    reg = DecisionTreeRegressor(random_state=1337, ccp_alpha=ccp_alpha)
    cv_results = cross_validate(reg, x_train, y_train, cv=K, scoring="neg_mean_squared_error", return_train_score=True, return_estimator=True)
    estimators = cv_results["estimator"]
    test_scores = -cv_results["test_score"]
    train_scores = -cv_results["train_score"]
    EPE.append(np.mean(test_scores))
    ETE.append(np.mean(train_scores))
    regs.append(estimators)



node_counts = [reg[0].tree_.node_count for reg in regs]
depth = [reg[0].tree_.max_depth for reg in regs]

fig = plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("Number of nodes in tree")
plt.savefig(fig_path("nodesvalpha.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
plt.xlabel(r'Cost complexity parameter $\alpha$')
plt.ylabel("Depth of tree")
plt.savefig(fig_path("depthvalpha.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, ETE, color="tab:blue", marker="o", label="ETE", drawstyle="steps-post")
plt.plot(ccp_alphas, EPE, color="tab:orange", marker="o", label="EPE", drawstyle="steps-post")
plt.vlines(x=ccp_alphas[np.argmin(EPE)], ymin=0, ymax=np.min(EPE), color="tab:orange", linestyle="dashed", label="Minimum EPE")
plt.xlabel(r"Cost complexity parameter $\alpha$")
plt.ylabel("MSE")
plt.legend()
plt.savefig(fig_path("CVRegTree.pdf"), format="pdf", bbox_inches="tight")

"""
Testing tree
"""
reg = DecisionTreeRegressor(random_state=1337, ccp_alpha=ccp_alphas[np.argmin(EPE)])
reg.fit(x_train, y_train)
preds_train = reg.predict(x_train)
preds = reg.predict(x_test)
TestMSE = mean_squared_error(y_test, preds)
TestR2 = r2_score(y_test, preds)
TrainMSE = mean_squared_error(y_train, preds_train)
TrainR2 = r2_score(y_train, preds_train)
print(f"Test: alpha={ccp_alphas[np.argmin(EPE)]}")
print(f"Train: MSE={TrainMSE}, R2={TrainR2}")
print(f"Test: MSE={TestMSE}, R2={TestR2}")

fig = plt.figure(figsize=(8, 5))
tree.plot_tree(reg, filled=True)
plt.savefig(fig_path("RegTree.pdf"), format="pdf", bbox_inches="tight")
