"""Common interface to sample solutions from Bayesian methods."""
import random

def bbb_sample_solutions(solver, n_samples, n_bundle_parameters):
    parameter_samples = solver.sample_posterior(n_samples)
    def sample_solutions(coords, *eq_params, to_numpy=False):
        eq_params = [eq_params[i].reshape(-1, 1) for i in range(n_bundle_parameters)]
        return solver.posterior_predictive([coords, *eq_params], param_samples=parameter_samples, to_numpy=to_numpy)
    return sample_solutions
        

def nlm_sample_solutions(nlm_model, n_samples, n_bundle_parameters):
    def sample_solutions(coords, *params, to_numpy=False):
        return nlm_model.sample_posterior_predictive([coords, *params[:n_bundle_parameters]], n_samples, to_numpy=to_numpy)
    return sample_solutions

def hmc_sample_solutions(solver, param_samples, n_samples, n_bundle_parameters):
    n_param_samples = list(param_samples.values())[0].shape[0]
    subsample_idx = random.sample(range(n_param_samples), min(n_param_samples, n_samples))
    subsamples = {k: v[subsample_idx] for k, v in param_samples.items()}
    def sample_solutions(coords, *params, to_numpy=False):
        if to_numpy:
            return solver.posterior_predictive([coords, *params[:n_bundle_parameters]], param_samples=subsamples, to_numpy=True)
        return solver.posterior_predictive([coords, *params[:n_bundle_parameters]], param_samples=subsamples)
    return sample_solutions