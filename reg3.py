import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

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
Loading data
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
iterations = 10
random_seeds = rng.integers(1e5, size=(iterations,))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
lr = LinearRegression()
bic_scores_f = []
aic_scores_f = []
bic_scores_b = []
aic_scores_b = []
for i in range(7):
    AIC_score = make_scorer(AIC, K=i+1, greater_is_better=False)
    BIC_score = make_scorer(BIC, K=i+1, greater_is_better=False)
    sfs_lr_forward = SequentialFeatureSelector(
    lr, n_features_to_select=i+1, direction="forward", n_jobs=5, scoring=AIC_score
    ).fit(x_train, y_train)
    x_aic_f_train = sfs_lr_forward.transform(x_train)
    x_aic_f_test = sfs_lr_forward.transform(x_test)
    sfs_lr_backward = SequentialFeatureSelector(
    lr, n_features_to_select=i+1, direction="backward", n_jobs=5, scoring=AIC_score
    ).fit(x_train, y_train)
    x_aic_b_train = sfs_lr_backward.transform(x_train)
    x_aic_b_test = sfs_lr_backward.transform(x_test)
    valid_indices_forward = [i for i in range(8) if sfs_lr_forward.get_support()[i]]
    valid_indices_backward = [i for i in range(8) if sfs_lr_backward.get_support()[i]]
    print("Features selected by forward sequential selection:")
    print(f"{[dependent_labels[valid_indices_forward[j]] for j in range(len(valid_indices_forward))]}")

    print("Features selected by backward sequential selection: ")
    print(f"{[dependent_labels[valid_indices_backward[j]] for j in range(len(valid_indices_backward))]}")

    sfs_lr_forward = SequentialFeatureSelector(
    lr, n_features_to_select=i+1, direction="forward", n_jobs=5, scoring=BIC_score
    ).fit(x_train, y_train)
    x_bic_f_train = sfs_lr_forward.transform(x_train)
    x_bic_f_test = sfs_lr_forward.transform(x_test)
    sfs_lr_backward = SequentialFeatureSelector(
    lr, n_features_to_select=i+1, direction="backward", n_jobs=5, scoring=BIC_score
    ).fit(x_train, y_train)
    x_bic_b_train = sfs_lr_backward.transform(x_train)
    x_bic_b_test = sfs_lr_backward.transform(x_test)
    valid_indices_forward = [i for i in range(8) if sfs_lr_forward.get_support()[i]]
    valid_indices_backward = [i for i in range(8) if sfs_lr_backward.get_support()[i]]
    print("Features selected by forward sequential selection:")
    print(f"{[dependent_labels[valid_indices_forward[j]] for j in range(len(valid_indices_forward))]}")

    print("Features selected by backward sequential selection: ")
    print(f"{[dependent_labels[valid_indices_backward[j]] for j in range(len(valid_indices_backward))]}")
    # Forward
    lr.fit(x_aic_f_train, y_train)
    preds = lr.predict(x_aic_f_test)
    aic_scores_f.append(AIC(y_test, preds, K=i+1))
    lr.fit(x_aic_b_train, y_train)
    preds = lr.predict(x_aic_b_test)
    aic_scores_b.append(AIC(y_test, preds, K=i+1))
    lr.fit(x_bic_f_train, y_train)
    preds = lr.predict(x_bic_f_test)
    bic_scores_f.append(BIC(y_test, preds, K=i+1))
    lr.fit(x_bic_b_train, y_train)
    preds = lr.predict(x_bic_b_test)
    bic_scores_b.append(BIC(y_test, preds, K=i+1))

lr.fit(x_train, y_train)
preds = lr.predict(x_test)
aic_scores_f.append(AIC(y_test, preds, K=8))
aic_scores_b.append(AIC(y_test, preds, K=8))
bic_scores_f.append(BIC(y_test, preds, K=8))
bic_scores_b.append(BIC(y_test, preds, K=8))
fig = plt.figure(figsize=(8, 5))
plt.plot([i+1 for i in range(8)], aic_scores_f, color="tab:blue", label="AIC forward")
plt.plot([i+1 for i in range(8)], aic_scores_b, color="tab:red", label="AIC backward")
plt.plot([i+1 for i in range(8)], bic_scores_f, color="tab:green", label="BIC forward")
plt.plot([i+1 for i in range(8)], bic_scores_b, color="tab:orange", label="BIC backward")
plt.xlabel("Number of features")
plt.ylabel("Score")
plt.vlines(x=np.argmin(aic_scores_f)+1, ymin=np.min(aic_scores_f), ymax=1095, color="tab:red",linestyle="dashed", label="Minimum AIC")
plt.vlines(x=np.argmin(bic_scores_f)+1, ymin=np.min(bic_scores_f), ymax=1095, color="tab:orange", linestyle="dashed", label="Minimum BIC")
plt.legend()
plt.savefig(fig_path("aicbicscores.pdf"), format="pdf", bbox_inches="tight")


"""
AIC error estimation
"""
subsetx = x[["TPSA", "SAacc", "MLOGP", "nN"]]

x_train, x_test, y_train, y_test = train_test_split(subsetx, y, test_size=0.33, random_state=seed)
lr = LinearRegression()
lr.fit(x_train, y_train)
train_preds = lr.predict(x_train)
MSE_train = mean_squared_error(y_train, train_preds)
r2_train = r2_score(y_train, train_preds)
preds = lr.predict(x_test)
MSE = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("AIC error on training set:")
print(f"MSE={MSE_train}, R2={r2_train}.")
print("AIC error on test set:")
print(f"MSE={MSE}, R2={r2}")

"""
BIC error estimation
"""
subsetx = x[["TPSA", "SAacc", "MLOGP"]]
x_train, x_test, y_train, y_test = train_test_split(subsetx, y, test_size=0.33, random_state=seed)
lr = LinearRegression()
lr.fit(x_train, y_train)
train_preds = lr.predict(x_train)
MSE_train = mean_squared_error(y_train, train_preds)
r2_train = r2_score(y_train, train_preds)
preds = lr.predict(x_test)
MSE = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("BIC error on training set:")
print(f"MSE={MSE_train}, R2={r2_train}.")
print("BIC error on test set:")
print(f"MSE={MSE}, R2={r2}")
