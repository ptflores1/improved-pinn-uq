import numpy as np
import pandas as pd
import torch

class CCDeterministic:
    """Cosmic Chronometers with deterministic Hubble parameter."""
    def __init__(self, params_min, params_max, solution, H_fn, dataset_path):
        self.params_min = params_min
        self.params_max = params_max
        self.solution = solution
        self.H_fn = H_fn
        self.load_data(dataset_path)

    def load_data(self, dataset_path):
        cc_df = pd.read_csv(dataset_path)
        self.cc_z = np.array(cc_df["z"].values).reshape(-1, 1)
        self.cc_h = np.array(cc_df["h"].values)
        self.cc_std = np.array(cc_df["std"].values)

    def log_prior(self, theta):
        for param, p_min, p_max in zip(theta, self.params_min, self.params_max):
            if not (p_min <= param <= p_max):
                return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        H = self.H_fn(self.cc_z, *theta, self.solution)
        assert H.size == self.cc_h.size
        return -.5*torch.tensor(((self.cc_h - H.ravel()) / self.cc_std) ** 2).sum()

    @torch.no_grad
    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll
    
class CCBayesian(CCDeterministic):
    """Cosmic Chronometers with Bayesian Hubble parameter."""
    def __init__(self, params_min, params_max, sample_solutions, H_fn, dataset_path):
        self.params_min = params_min
        self.params_max = params_max
        self.sample_solutions = sample_solutions
        self.H_fn = H_fn
        self.load_data(dataset_path)

    def log_likelihood(self, theta):
        H = self.H_fn(self.cc_z, *theta, self.sample_solutions)
        assert H.shape[1:] == self.cc_h.reshape(-1, 1).shape
        nan_prop = np.sum(np.isnan(H)) / H.size
        return -.5*torch.tensor(((self.cc_h.reshape(-1, 1) - H) / self.cc_std.reshape(-1, 1)) ** 2).nansum() * (1 - nan_prop)