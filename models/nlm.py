import io
import torch
import pickle
from typing import Callable, List, Sequence
from tqdm.auto import tqdm
from models.nets import FCNN

eps = torch.sqrt(torch.tensor(torch.finfo().eps)).item()

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            if not torch.cuda.is_available():
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return lambda b: torch.load(io.BytesIO(b))
        else: return super().find_class(module, name)

class NLMModel:
    @staticmethod
    def load(path, device=None):
        with open(path, 'rb') as f:
            loaded = CPU_Unpickler(f).load()
        if device is not None:
            loaded.device = device
        return loaded
        
    @staticmethod
    def _compute_sigma_post(Phi, Sigma_e, sigma_prior):
        return torch.linalg.inv(Phi.T.matmul(torch.linalg.inv(Sigma_e)).matmul(
            Phi) + (sigma_prior**(-2))*torch.eye(Phi.size(1), device=Phi.device))

    @staticmethod
    def _compute_mu_post(Sigma_post, Sigma_e, Phi, u):
        return Sigma_post.matmul((torch.linalg.inv(Sigma_e).matmul(Phi)).T).matmul(u).ravel()

    @staticmethod
    def _posterior(Phis, Sigmas_e, sigma_priors, us):
        Sigmas_post = [NLMModel._compute_sigma_post(Phi, Sigma_e, sigma_prior)
                       for Phi, Sigma_e, sigma_prior in zip(Phis, Sigmas_e, sigma_priors)]
        mus_posts = [NLMModel._compute_mu_post(Sigma_post, Sigma_e, Phi, u)
                     for Sigma_post, Sigma_e, Phi, u in zip(Sigmas_post, Sigmas_e, Phis, us)]
        return mus_posts, Sigmas_post

    @staticmethod
    def _posterior_predictive(mu_posts, Sigma_posts, Phis_test, Sigmas_e_test, include_Sigmas_e=True):
        mu_postpreds = [Phi_test.matmul(mu_post).unsqueeze(-1) for Phi_test, mu_post in zip(Phis_test, mu_posts)]
        Sigma_postpreds = [
            Sigma_e_test*int(include_Sigmas_e) + Phi_test.matmul(Sigma_post).matmul(Phi_test.T) for Sigma_post, Phi_test,
            Sigma_e_test in zip(Sigma_posts, Phis_test, Sigmas_e_test)]
        return mu_postpreds, Sigma_postpreds

    def __init__(self, det_nets: List[FCNN], det_solutions: Callable, get_likelihood_stds: Callable, diff_eqs, conditions=None, device="cpu") -> None:
        self.det_solutions = det_solutions
        self.det_nets = det_nets
        self.get_likelihood_stds = get_likelihood_stds
        self.conditions = conditions
        self.device = device
        self.diff_eqs = diff_eqs

    def get_feature_basis(self, coordinates) -> List[torch.Tensor]:
        coordinates = torch.cat([torch.tensor(c, device=self.device) for c in coordinates], dim=1)
        fmaps = [net.feature_map(coordinates).detach() for net in self.det_nets]
        ones = [torch.ones((fmaps[0].size(0), 1), device=self.device) for _ in self.det_nets]
        Phis = [torch.cat((one, fmap), axis=1) for one, fmap in zip(ones, fmaps)]
        return Phis

    def get_phi_and_sigma(self, coordinates):
        Phis = self.get_feature_basis(coordinates)
        diagonals = [torch.clamp(stds.to(self.device).ravel()**2, min=eps) for stds in self.get_likelihood_stds(coordinates)]
        Sigmas_e = [torch.diag(diagonal) for diagonal in diagonals]
        return Phis, Sigmas_e

    def fit(self, sigma_priors: float, coordinates):
        Phis, Sigmas_e = self.get_phi_and_sigma(coordinates)
        solutions = self.det_solutions(*coordinates)
        u_nets = solutions if isinstance(solutions, list) else [solutions]
        self.mu_posts, self.Sigma_posts = NLMModel._posterior(Phis, Sigmas_e, sigma_priors, u_nets)

    def posterior_predictive(self, coordinates, include_Sigmas_e=True):
        Phis, Sigmas_e = self.get_phi_and_sigma(coordinates)
        mu_postpreds, Sigma_postpreds = NLMModel._posterior_predictive(self.mu_posts, self.Sigma_posts, Phis, Sigmas_e, include_Sigmas_e)
        sigmas_postpred = [torch.sqrt(Sigma_postpred.diagonal()).ravel() for Sigma_postpred in Sigma_postpreds]
        return mu_postpreds, sigmas_postpred

    def sample_posterior_predictive(self, coordinates, n_samples, to_numpy=False):
        mu_postpreds, sigmas_postpred = self.posterior_predictive(coordinates)
        samples = [torch.distributions.Normal(mu_postpred, sigma_postpred.reshape_as(mu_postpred)).sample((n_samples,)) for mu_postpred, sigma_postpred in zip(mu_postpreds, sigmas_postpred)]
        if to_numpy:
            return [s.cpu().numpy() for s in samples]
        return samples
    
    def get_residuals(self, coordinates):
        coordinates = [torch.tensor(c).requires_grad_(True) for c in coordinates]
        mu, _ = self.posterior_predictive(coordinates)
        residuals = self.diff_eqs(*mu, *coordinates)
        return residuals

    def calibrate_prior(self, prior_range: Sequence[float], test_coordinates, train_coordinates):
        Phis, Sigmas_e = self.get_phi_and_sigma(train_coordinates)

        solutions = self.det_solutions(*train_coordinates)
        us = solutions if isinstance(solutions, list) else [solutions]

        Phis_test, Sigmas_e_test = self.get_phi_and_sigma(test_coordinates)
        us_test = self.det_solutions(*test_coordinates)

        error_norms = [[] for _ in range(len(us))]
        valid_priors = [[] for _ in range(len(us))]
        valid_error_norms = [[] for _ in range(len(us))]

        for prior in tqdm(prior_range):
            # calculate posterior predictive
            mus_post, Sigmas_post = NLMModel._posterior(Phis, Sigmas_e, [prior] * len(us), us)
            mus_postpred, Sigmas_postpred = NLMModel._posterior_predictive(
                mus_post, Sigmas_post, Phis_test, Sigmas_e_test)

            # calculate std of Sigma_e_test and Sigma_postpred
            stds_e_test = [torch.sqrt(Sigma_e_test.diagonal()).ravel() for Sigma_e_test in Sigmas_e_test]
            stds_postpred = [torch.sqrt(Sigma_postpred.diagonal()).ravel() for Sigma_postpred in Sigmas_postpred]
            # for all priors
            error_means = [torch.abs(mu_postpred - u_test) for mu_postpred, u_test in zip(mus_postpred, us_test)]
            error_stds = [torch.abs(std_postpred - std_e_test)
                          for std_postpred, std_e_test in zip(stds_postpred, stds_e_test)]
            funcs_error_norms = [(torch.sum(error_mean) + torch.sum(error_std)).item()
                                 for error_mean, error_std in zip(error_means, error_stds)]
            for func_idx in range(len(us)):
                error_norms[func_idx].append(funcs_error_norms[func_idx])
                error_mean, std_postpred, std_e_test, error_std = error_means[
                    func_idx], stds_postpred[func_idx], stds_e_test[func_idx], error_stds[func_idx]
                # find valid priors
                count = []
                for i in range(test_coordinates[0].shape[0]):
                    if error_mean[i] <= 3*std_postpred[i] - std_e_test[i]:
                        count.append(i)
                total = len(count)
                if total == test_coordinates[0].shape[0]:
                    valid_priors[func_idx].append(prior)
                    # for valid priors
                    valid_error_norm = (torch.sum(error_mean) + torch.sum(error_std)).item()
                    valid_error_norms[func_idx].append(valid_error_norm)

        prior_opts = []
        for func_idx in range(len(us)):
            # find prior that minimizes ||mu_postpred - u_test|| + ||std_postpred - std_e_test||
            if len(valid_priors[func_idx]) == 0:
                prior_opt = None
            else:
                idx = valid_error_norms[func_idx].index(min(valid_error_norms[func_idx]))
                prior_opt = valid_priors[func_idx][idx].item()
            prior_opts.append(prior_opt)
        return valid_priors, prior_opts, error_norms
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
