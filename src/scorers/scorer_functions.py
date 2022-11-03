import numpy as np

def RSS(true, preds):
    return np.sum((true-preds)**2)

def AIC(true, preds, K=1):
    rss = RSS(true, preds)
    N = len(true)
    return 2*K + N*np.log(rss)


def BIC(true, preds, K=1):
    rss = RSS(true, preds)
    N = len(true)
    return np.log(N)*K + N*np.log(rss)
