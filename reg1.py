import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from src import *
from utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from sklearn.metrics import make_scorer, mean_squared_error, r2_score


def dichotomize(df, colnames):
    dfDich = df.copy()
    for colname in colnames:
      dfDich[colname] = np.where(dfDich[colname]>0, 1, 0)
    return dfDich

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
#xD_train, xD_test, y_train, y_test = train_test_split(xD, y, test_size=0.33, random_state=seed)
xD_train = dichotomize(x_train, dichotomize_labels)
xD_test = dichotomize(x_test, dichotomize_labels)

lr = LinearRegression()
lr.fit(x_train, y_train)
preds_train = lr.predict(x_train)
preds = lr.predict(x_test)
MSEtrain_nd = mean_squared_error(y_train, preds_train)
r2train_nd = r2_score(y_train, preds_train)
MSEtest_nd = mean_squared_error(y_test, preds)
r2test_nd = r2_score(y_test, preds)

lr = LinearRegression()
lr.fit(xD_train, y_train)
preds_train = lr.predict(xD_train)
preds = lr.predict(xD_test)
MSEtrain_d = mean_squared_error(y_train, preds_train)
r2train_d = r2_score(y_train, preds_train)
MSEtest_d = mean_squared_error(y_test, preds)
r2test_d = r2_score(y_test, preds)

print("Training errors:")
print(f"ND: MSE={MSEtrain_nd}, R2={r2train_nd}")
print(f"D: MSE={MSEtrain_d}, R2={r2train_d}")
print("Testing errors:")
print(f"ND: MSE={MSEtest_nd}, R2={r2test_nd}")
print(f"D: MSE={MSEtest_d}, R2={r2test_d}")
