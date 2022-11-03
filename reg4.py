import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
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
seed = 2959
rng = default_rng(seed)
x = qsar.x.to_numpy() # non-dichotomized
xD = qsar.xdich # dichotomized
y = qsar.y.to_numpy() # response

"""
Bootstrap resampling
"""


sss = ShuffleSplit(n_splits=1, test_size=0.33)
sss.get_n_splits(x, y)
train_indices, test_indices = next(sss.split(x, y))
x_train, x_test =x[train_indices], x[test_indices]
y_train, y_test =y[train_indices], y[test_indices]
nsamples = x_train.shape[0]
sampler = Sampler(nsamples, train_indices)
B = 1000 # using 10000 bootstrap samples
nvals = 51
random_seeds = rng.integers(1e5, size=(B))
logalphas = np.linspace(-5, 5, nvals)
alphas = 10**logalphas
Bscores_test = np.zeros(nvals)
Bscores_train = np.zeros(nvals)
Bcoeffs = np.zeros((nvals, x_train.shape[1]+1))
for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    b_errors_train = np.zeros(B)
    b_errors_test = np.zeros(B)
    coefficients = np.zeros((B, x_train.shape[1]+1))
    for b in range(B):
        x_tr, x_te, y_tr, y_te = sampler.sample(x_train, y_train, random_seeds[b])
        ridge.fit(x_tr, y_tr)
        preds = ridge.predict(x_te)
        b_errors_test = np.mean((y_te - preds)**2) # MSE
        b_errors_train = np.mean((y_tr - ridge.predict(x_tr))**2)
        coefficients[b, 0] = ridge.intercept_
        coefficients[b, 1:] = ridge.coef_

    Bscores_test[i] = np.mean(b_errors_test)
    Bscores_train[i] = np.mean(b_errors_train)
    Bcoeffs[i, :] = np.mean(coefficients, axis=0)



"""
K-fold cross validation
"""
K = 10
CVscores_test = np.zeros(nvals)
CVscores_train = np.zeros(nvals)
CVcoeffs = np.zeros((nvals, x_train.shape[1]+1))
for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha)
    cv_results = cross_validate(ridge, x_train, y_train, cv=K, scoring="neg_mean_squared_error", return_train_score=True, return_estimator=True)
    estimators = cv_results["estimator"]
    intercept = 0
    coefficients = np.zeros(x_train.shape[1])
    for model in estimators:
        intercept += model.intercept_
        coefficients += model.coef_
    coefficients = coefficients / K
    intercept = intercept / K
    CVcoeffs[i, 0] = intercept
    CVcoeffs[i, 1:] = coefficients
    CVscores_test[i] = -np.mean(cv_results["test_score"])
    CVscores_train[i] = -np.mean(cv_results["train_score"])


"""
Plots
"""


fig = plt.figure(figsize=(8, 5))
for i in range(Bcoeffs.shape[1]):
    plt.plot(logalphas, Bcoeffs[:, i], label=f"{labels_wI[i]}")
plt.xlabel(r"$\mathrm{log}(\alpha)$")
plt.ylabel("Coefficient value")
plt.legend()
plt.savefig(fig_path("BScoefficients.pdf"), format="pdf", bbox_inches="tight")


fig = plt.figure(figsize=(8, 5))
for i in range(CVcoeffs.shape[1]):
    plt.plot(logalphas, CVcoeffs[:, i], label=f"{labels_wI[i]}")
plt.ylabel("Coefficient value")
plt.xlabel(r"$\mathrm{log}(\alpha)$")
plt.legend()
plt.savefig(fig_path("CVcoefficients.pdf"), format="pdf", bbox_inches="tight")

fig = plt.figure(figsize=(8,5))
plt.plot(logalphas, Bscores_test, color="tab:red", label="EPE Bootstrap")
plt.plot(logalphas, CVscores_test, color="tab:orange", label=f"EPE {K}-fold CV")
plt.vlines(x=logalphas[np.argmin(Bscores_test)], ymin=np.min(Bscores_test), ymax=2.8, color="tab:red", linestyle="dashed", label="Bootstrap minima")
plt.vlines(x=logalphas[np.argmin(CVscores_test)], ymin=np.min(CVscores_test), ymax=2.8, color="tab:orange", linestyle="dashed", label="CV minima")
plt.xlabel(r"$\mathrm{log}(\alpha)$")
plt.ylabel("EPE (MSE)")
plt.legend()
plt.savefig(fig_path("test_error_ridgeCVBS.pdf"), format="pdf", bbox_inches="tight")


fig = plt.figure(figsize=(8,5))
plt.plot(logalphas, Bscores_train, label="ETE Bootstrap")
plt.plot(logalphas, CVscores_train, label=f"ETE {K}-fold CV")
plt.xlabel(r"$\mathrm{log}(\alpha)$")
plt.ylabel("ETE (MSE)")
plt.legend()
plt.savefig(fig_path("train_error_ridgeCVBS.pdf"), format="pdf", bbox_inches="tight")

"""
Testing best alphas
"""
# Bootstrap best
ridge = Ridge(alpha=alphas[np.argmin(Bscores_test)])
ridge.fit(x_train, y_train)
bootstrap_train_preds = ridge.predict(x_train)
bootstrap_preds = ridge.predict(x_test)
MSE_trainB = mean_squared_error(y_train, bootstrap_train_preds)
r2_trainB = r2_score(y_train, bootstrap_train_preds)
MSE_B = mean_squared_error(y_test, bootstrap_preds)
r2_B = r2_score(y_test, bootstrap_preds)


ridge = Ridge(alpha=alphas[np.argmin(CVscores_test)])
ridge.fit(x_train, y_train)
CV_train_preds = ridge.predict(x_train)
CV_preds = ridge.predict(x_test)
MSE_trainCV = mean_squared_error(y_train, CV_train_preds)
r2_trainCV = r2_score(y_train, CV_train_preds)
MSE_CV = mean_squared_error(y_test, CV_preds)
r2_CV = r2_score(y_test, CV_preds)

print(f"Best alphaCV={alphas[np.argmin(CVscores_test)]}")
print(f"Best alphaBS={alphas[np.argmin(Bscores_test)]}")
print("Training errors: ")
print(f"Bootstrap: MSE={MSE_trainB}, R2={r2_trainB}")
print(f"CV: MSE={MSE_trainCV}, R2={r2_trainCV}")
print("Errors on test sets: ")
print(f"Bootstrap: MSE={MSE_B}, R2={r2_B}")
print(f"CV: MSE={MSE_CV}, R2={r2_CV}")
