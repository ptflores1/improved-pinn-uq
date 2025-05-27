import os
import dill
import torch
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct

from neurodiffeq.solvers import Solver1D

from models.nlm import NLMModel

def decorate_dill(method):
    def wrapper(func):
        def wrapped_f(*args, **kwargs):
            force = kwargs.get("force", False)
            print("force", force)
            final_file_path = args[-1] + f"_{method}.dill"
            if os.path.exists(final_file_path) and not force:
                print("Loading plot data from file", final_file_path)
                with open(final_file_path, "rb") as f:
                    return dill.load(f)
            data = func(*args, **kwargs)
            print("Saving plot data to file", final_file_path)
            with open(final_file_path, "wb") as f:
                dill.dump(data, f)
            return data
        return wrapped_f
    return wrapper

@decorate_dill("fcnn")
def get_fcnn_data(model_name, coords, filename_prefix, force=False):
    print("Loading FCNN")
    solver = Solver1D.load(model_name)
    solution = solver.get_solution()(coords, to_numpy=True)
    data = { "FCNN": solution }
    return data

@decorate_dill("bbb")
def get_bbb_data(model_name, coords, filename_prefix, force=False):
    print("Loading BBB")
    solver_bbb = torch.load(model_name, map_location="cpu")
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples, bbb_residuals = solver_bbb.posterior_predictive([torch.tensor(coords)], num_samples=10_000, to_numpy=True, include_residuals=True)
    data = {
         "BBB": (bbb_samples[0].mean(axis=0), bbb_samples[0].std(axis=0)),
         "BBB_samples": bbb_samples,
         "BBB_residuals": bbb_residuals,
         }
    return data

@decorate_dill("nlm")
def get_nlm_data(model_name, coords, filename_prefix, force=False):
    print("Loading NLM")
    nlm_model = NLMModel.load(model_name)
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_mean, nlm_std = nlm_model.posterior_predictive([coords], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([coords], n_samples=10_000, to_numpy=True)
    data = {
         "NLM": (nlm_mean[0].detach().cpu(), nlm_std[0].detach().cpu()),
         "NLM_samples": nlm_samples,
         }
    return data

@decorate_dill("hmc")
def get_hmc_data(model_name, posterior_samples_file, coords, filename_prefix, force=False):
    print("Loading HMC")
    hmc_solver = torch.load(model_name, pickle_module=dill, map_location="cpu")
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(posterior_samples_file, pickle_module=dill, map_location="cpu")
    hmc_samples = hmc_solver.posterior_predictive([torch.tensor(coords)], hmc_posterior_samples, to_numpy=True)
    data = {
         "HMC": (hmc_samples[0].mean(axis=0), hmc_samples[0].std(axis=0)),
         "HMC_samples": hmc_samples,
         }
    return data


def plot_calibration_area(y_preds, y_stds, y_trues, labels, ax):
    for y_pred, y_std, y_true, label in zip(y_preds, y_stds, y_trues, labels):
        (exp_proportions, obs_proportions) = uct.metrics_calibration.get_proportion_lists_vectorized(y_pred, y_std, y_true, prop_type="quantile")
        miscalibration_area = uct.metrics_calibration.miscalibration_area_from_proportions(exp_proportions=exp_proportions, obs_proportions=obs_proportions)
        print(label, miscalibration_area)
        ax.plot(exp_proportions, obs_proportions, label=label)
    
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="black")
    ax.set_xlabel("Predicted Proportion in Interval")
    ax.set_ylabel("Observed Proportion in Interval")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration")
    #ax.legend(prop={'size': 10})

    return ax