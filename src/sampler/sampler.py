import pandas as pd
from numpy.random import default_rng
import numpy as np


class Sampler:
    def __init__(self, nsamples, train_indices, rng=None):
        self.nsamples = nsamples
        if rng==None:
            rng = default_rng
        else:
            rng = rng

        self.rng = rng
        self.train_indices = train_indices


    def resample(self, seed):
        rng = self.rng(seed)
        sample_indices = rng.integers(self.nsamples, size=self.nsamples)
        not_in_sample_indices = []
        for idx in range(self.nsamples):
            if np.any(sample_indices == idx):
                continue
            else:
                not_in_sample_indices.append(idx)

        not_in_sample_indices = np.array(not_in_sample_indices)
        return sample_indices, not_in_sample_indices

    def sample(self, x, y, seed):
        in_sample, outside_sample = self.resample(seed)
        x_tr = x[in_sample]
        y_tr = y[in_sample]
        x_te = x[outside_sample]
        y_te = y[outside_sample]
        return x_tr, x_te, y_tr, y_te
