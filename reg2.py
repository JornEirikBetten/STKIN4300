import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats

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
iterations = 200
random_seeds = rng.integers(1e5, size=(iterations,))

# split range: test sizes [0.01, 0.99]
test_sizes = np.linspace(0.1, 0.6, 200)
training_errors_d = np.zeros((iterations, 200))
test_errors_d = np.zeros((iterations, 200))
training_errors_nd = np.zeros((iterations, 200))
test_errors_nd = np.zeros((iterations, 200))
plt.figure(figsize=(8,5))
for j in range(iterations):
    for i, test_size in enumerate(test_sizes):
        # split data correspondingly
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seeds[j])
        xD_train, xD_test, y_train, y_test = train_test_split(xD, y, test_size=test_size, random_state=random_seeds[j])
        model_nd = LinearModel(x_train, y_train, labels_wI)
        model_d = LinearModel(xD_train, y_train, labels_wI)

        preds_train_nd = model_nd.predict(x_train)
        training_errors_nd[j,i] = model_nd.MSE(y_train, preds_train_nd)
        preds_test_nd = model_nd.predict(x_test)
        test_errors_nd[j,i] = model_nd.MSE(y_test, preds_test_nd)
        preds_train_d = model_d.predict(xD_train)
        training_errors_d[j,i] = model_d.MSE(y_train, preds_train_nd)
        preds_test_d = model_d.predict(xD_test)
        test_errors_d[j,i] = model_d.MSE(y_test, preds_test_d)

    plt.plot(test_sizes, test_errors_nd[j, :], color="tab:blue", alpha=0.05)
    plt.plot(test_sizes, test_errors_d[j, :], color="tab:red", alpha=0.05)

plt.plot(test_sizes, np.mean(test_errors_nd, axis=0), color="tab:blue", label="$\sim\mathrm{E}_{\mathrm{err}, \mathrm{test}}[\mathcal{D}_{ND}]$")
plt.plot(test_sizes, np.mean(test_errors_d, axis=0), color="tab:red", label="$\sim\mathrm{E}_{\mathrm{err}, \mathrm{test}}[\mathcal{D}_D]$")
plt.xlabel("Relative size of test set")
plt.ylabel("Mean squared error")
plt.legend()
plt.savefig(fig_path("test_errors2.pdf"), format="pdf", bbox_inches='tight')

fig = plt.figure(figsize=(8, 5))
for j in range(iterations):
    plt.plot(test_sizes, training_errors_nd[j, :], color="tab:blue", alpha=0.05)
    plt.plot(test_sizes, training_errors_d[j, :], color="tab:red", alpha=0.05)

plt.plot(test_sizes, np.mean(training_errors_nd, axis=0), color="tab:blue", label="$\sim\mathrm{E}_{\mathrm{err}, \mathrm{train}}[\mathcal{D}_{ND}]$")
plt.plot(test_sizes, np.mean(training_errors_d, axis=0), color="tab:red", label="$\sim\mathrm{E}_{\mathrm{err}, \mathrm{train}}[\mathcal{D}_{D}]$")
plt.xlabel("Relative size of test set")
plt.ylabel("Mean squared error")
plt.legend()
plt.savefig(fig_path("train_errors2.pdf"), format="pdf", bbox_inches="tight")
