import numpy as np
import pandas as pd
import torch

class LABDeterministic:
    def __init__(self, params_min, params_max, solution, dataset_path):
        self.params_min = params_min
        self.params_max = params_max
        self.solution = solution
        self.load_data(dataset_path)

    def load_data(self, dataset_path):
        lab_df = pd.read_csv(dataset_path)
        self.lab_t = np.array(lab_df["t"].values).reshape(-1, 1)
        self.lab_x = np.array(lab_df["x"].values)
        self.lab_x_std = np.array(lab_df["x"].values).std()
        self.lab_y = np.array(lab_df["y"].values)
        self.lab_y_std = np.array(lab_df["y"].values).std()
        self.lab_z = np.array(lab_df["z"].values)
        self.lab_z_std = np.array(lab_df["z"].values).std()

    def log_prior(self, theta):
        for param, p_min, p_max in zip(theta, self.params_min, self.params_max):
            if not (p_min <= param <= p_max):
                return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        thetas = [t*torch.ones(self.lab_t.shape).cpu().numpy() for t in theta]
        x, y, z = self.solution(self.lab_t, *thetas)
        x_log_prob = -.5*np.sum(((self.lab_x.ravel() - x.cpu().detach().numpy().ravel()) / self.lab_x_std) ** 2)
        y_log_prob = -.5*np.sum(((self.lab_y.ravel() - y.cpu().detach().numpy().ravel()) / self.lab_y_std) ** 2)
        z_log_prob = -.5*np.sum(((self.lab_z.ravel() - z.cpu().detach().numpy().ravel()) / self.lab_z_std) ** 2)
        return x_log_prob + y_log_prob + z_log_prob

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll
    
class LABBayesian(LABDeterministic):
    def __init__(self, params_min, params_max, sample_solutions, dataset_path):
        self.params_min = params_min
        self.params_max = params_max
        self.sample_solutions = sample_solutions
        self.load_data(dataset_path)

    def log_likelihood(self, theta):
        thetas = [p*torch.ones(self.lab_t.shape) for p in theta]
        x, y, z = self.sample_solutions(self.lab_t, *thetas)
        x_log_prob = -.5*np.sum(((self.lab_x.reshape(-1, 1) - x.cpu().detach().numpy()) / self.lab_x_std) ** 2)
        y_log_prob = -.5*np.sum(((self.lab_y.reshape(-1, 1) - y.cpu().detach().numpy()) / self.lab_y_std) ** 2)
        z_log_prob = -.5*np.sum(((self.lab_z.reshape(-1, 1) - z.cpu().detach().numpy()) / self.lab_z_std) ** 2)
        return x_log_prob + y_log_prob + z_log_prob