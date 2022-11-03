import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np





class LinearModel:
    def __init__(self, X, y, labels):
        self.lr = LinearRegression()
        self.X = X
        self.y = y
        self.labels = labels


    def __call__(self, X, y):
        self.model = self.lr.fit(X, y)
        return self.model

    def predict(self, X_test):
        model = self(self.X, self.y)
        predictions = model.predict(X_test)
        return predictions

    def MSE(self, true, predictions):
        return mean_squared_error(true, predictions)

    def r2(self, true, predictions):
        return r2_score(true, predictions)

    def OLS_sample(self, df):
        """
        Samples from the training set with replacement.
        """
        sample = df.sample(df.shape[0], replace=True)
        X_tr = sample[[c for c in sample.columns if c != 'y']]
        y_tr = sample.y

        lr = LinearRegression()
        lr.fit(X_tr, y_tr)

        params = [lr.intercept_] +  list(lr.coef_)

        return params

    def summary(self):
        """
        Makes a summary of significance of the coefficients of the
        linear model.
        """

        X = self.X
        y = self.y
        lr = LinearRegression()
        lr.fit(X, y)

        df = pd.DataFrame(X)
        df['y'] = y


        r_df = pd.DataFrame([self.OLS_sample(df) for _ in range(100)])

        w = [lr.intercept_] + list(lr.coef_)
        se = r_df.std()

        dof = X.shape[0] - X.shape[1] - 1

        summary = pd.DataFrame({
            'label':self.labels,
            'w': w,
            'se': se,
            't': w / se,
            '.025': w - se,
            '.975': w + se,
            'df': [dof for _ in range(len(w))]
        })

        summary['P>|t|'] = scipy.stats.t.sf(abs(summary.t), df=summary.df)

        return summary
