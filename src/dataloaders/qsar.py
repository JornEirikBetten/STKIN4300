import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


class QSAR:
  def __init__(self, path, dichotomize_labels):
    self.df = pd.read_csv(path)
    self.df_dich = self.dichotomize(dichotomize_labels)
    self.y = self.df["LC50"]
    self.xdich = self.df_dich.drop(["LC50"], axis=1)
    self.x = self.df.drop(["LC50"], axis=1)

  def dichotomize(self, colnames):
      dfDich = self.df.copy()
      for colname in colnames:
        dfDich[colname] = np.where(dfDich[colname]>0, 1, 0)
      return dfDich



if __name__ == "__main__":
  path = os.getcwd()
  filename = "/data/qsar_aquatic_toxicity.csv"
  qsar = QSAR(path+filename)
  print(qsar.df)
  print(qsar.dichotomize(["H-050", "C-040"]))
