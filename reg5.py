import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
import pandas as pd
import os
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from pygam import LinearGAM, PoissonGAM, s, te
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

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

order = 3
nsplines=8

"""
Split in train and test
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)
maxval_H050 = x["H-050"].max()
maxval_C040 = x["C-040"].max()
maxval_nN = x["nN"].max()

x = qsar.x.to_numpy() # non-dichotomized
xD = qsar.xdich # dichotomized
y = qsar.y.to_numpy() # response

sss = ShuffleSplit(n_splits=1, test_size=0.33)
sss.get_n_splits(x, y)
train_indices, test_indices = next(sss.split(x, y))
x_train, x_test =x[train_indices], x[test_indices]
y_train, y_test =y[train_indices], y[test_indices]
nsamples = x_train.shape[0]
sampler = Sampler(nsamples, train_indices)
B = 100 # of bootstrap samples
random_seeds = rng.integers(1e5, size=(B))
max_order = 3
BMSE_test = np.zeros(max_order)
BMSE_train = np.zeros(max_order)
BR2_test = np.zeros(max_order)
BR2_train = np.zeros(max_order)
for i in range(max_order):
    order = i+1
    n_splines = 9 + order
    print(f"Order = {order}")
    gam = LinearGAM(s(0,n_splines=nsplines, spline_order=order) + s(1, n_splines=nsplines, spline_order=order)\
                    +s(2, n_splines=nsplines, spline_order=order, dtype="categorical", edge_knots=[0, maxval_H050])\
                    +s(3,n_splines=nsplines, spline_order=order) + s(4,n_splines=nsplines, spline_order=order)+s(5,n_splines=nsplines, spline_order=order)\
                    +s(6,n_splines=nsplines, spline_order=order, dtype="categorical", edge_knots=[0, maxval_nN])\
                    +s(7,n_splines=nsplines, spline_order=order, dtype="categorical",edge_knots=[0, maxval_C040]), fit_intercept=True)
    b_errors_train = np.zeros(B)
    b_errors_test = np.zeros(B)
    b_r2_train = np.zeros(B)
    b_r2_test = np.zeros(B)
    for b in range(B):
        x_tr, x_te, y_tr, y_te = sampler.sample(x_train, y_train, random_seeds[b])
        gam.fit(x_tr, y_tr)
        preds_train = gam.predict(x_tr)
        preds = gam.predict(x_te)
        b_errors_test = mean_squared_error(y_te, preds)
        b_errors_train = mean_squared_error(y_tr, preds_train)
        b_r2_train = r2_score(y_tr, preds_train)
        b_r2_test = r2_score(y_te, preds)

    BMSE_test[i] = np.mean(b_errors_test)
    BMSE_train[i] = np.mean(b_errors_train)
    BR2_test[i] = np.mean(b_r2_test)
    BR2_train[i] = np.mean(b_r2_train)
fig = plt.figure(figsize=(8,5))
plt.plot([i+1 for i in range(max_order)], BMSE_test, label="EPE")
plt.plot([i+1 for i in range(max_order)], BMSE_train, label="ETE")
plt.xlabel("Order of smoothing splines")
plt.ylabel("MSE error")
plt.xticks([i+1 for i in range(max_order)], [f"{i+1}" for i in range(max_order)])
plt.legend()
plt.savefig(fig_path("GAM_BS_error_estimates.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot([i+1 for i in range(max_order)], BR2_test, label=r"$\mathrm{E}_{\mathrm{test, err}}[\mathrm{R}^2]$")
plt.plot([i+1 for i in range(max_order)], BR2_train, label=r"$\mathrm{E}_{\mathrm{train, err}}[\mathrm{R}^2]$")
plt.xlabel("Order of smoothing splines")
plt.ylabel("R2 score")
plt.legend()
plt.savefig(fig_path("GAM_BS_R2_estimates.pdf"), format="pdf", bbox_inches="tight")

for i in range(3):
    order = i+1
    n_splines = 9 + order
    gam = LinearGAM(s(0,n_splines=nsplines, spline_order=order) + s(1, n_splines=nsplines, spline_order=order)\
                    +s(2, n_splines=nsplines, spline_order=order, dtype="categorical", edge_knots=[0, maxval_H050])\
                    +s(3,n_splines=nsplines, spline_order=order) + s(4,n_splines=nsplines, spline_order=order)+s(5,n_splines=nsplines, spline_order=order)\
                    +s(6,n_splines=nsplines, spline_order=order, dtype="categorical", edge_knots=[0, maxval_nN])\
                    +s(7,n_splines=nsplines, spline_order=order, dtype="categorical",edge_knots=[0, maxval_C040]), fit_intercept=True)
    gam.fit(x_train, y_train)
    preds_train = gam.predict(x_train)
    preds = gam.predict(x_test)
    MSETestBest = mean_squared_error(y_test, preds)
    R2TestBest = r2_score(y_test, preds)
    MSETrainBest = mean_squared_error(y_train, preds_train)
    R2TrainBest = r2_score(y_train, preds_train)


    print(f"Order={order} of splines:")
    print(f"Train metrics: MSE={MSETrainBest}, R2={R2TrainBest}")
    print(f"Test metrics: MSE={MSETestBest}, R2={R2TestBest}")
