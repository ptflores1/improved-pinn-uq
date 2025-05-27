import numpy as np
from tqdm.auto import tqdm
import dill
import matplotlib.pyplot as plt
from neurodiffeq.solvers import Solver1D, BundleSolver1D
import torch
import equations.cpl_params as cpl
from models.nlm import NLMModel
from models.utils import StdGetterEB
from plotters.utils import dill_dec, dill_dec_old
from inverse.bayesian_adapters import nlm_sample_solutions, bbb_sample_solutions, hmc_sample_solutions
from .config import *
import uncertainty_toolbox as uct
from .utils import error_metrics

error_names = {
    "re": "Relative Error",
    "ae": "Absolute Error",
    "rpd": "Relative Percent Difference",
    "std_re": "Std. Relative Error",
    "std_ae": "Std. Absolute Error",
    "std_rpd": "Std. Relative Percent Difference",
}

def analytic_cpl(z):
    exp = 3*(np.log(z+1) + cpl.w_1*(np.log(z+1) + 1/(1+z) - 1) + cpl.w_0*np.log(z+1))
    return np.exp(exp)

def analytic_cpl_bundle(z, w_0, w_1):
    exp = 3*(np.log(z+1) + w_1*(np.log(z+1) + 1/(1+z) - 1) + w_0*np.log(z+1))
    return np.exp(exp)

@dill_dec_old("plot_data/cpl_params_forward.dill","plot_data/cpl_params_eb_forward.dill")
def get_plot_data(eb=False):
    eb = "_eb" if eb else ""
    x_test = np.linspace(cpl.coords_test_min, cpl.coords_test_max, 200).reshape(-1, 1)
    analytic = analytic_cpl(x_test)

    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = Solver1D.load("checkpoints/solver_cpl_params_fcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_cpl_params_bbb{eb}.pyro", map_location="cpu")
    solver_bbb.diff_eqs = cpl.system
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x_test)], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp[0])
        bbb_residuals.append(bbb_residuals_tmp[0])
    bbb_samples = np.concatenate(bbb_samples, axis=0)
    bbb_residuals = np.concatenate(bbb_residuals, axis=0)

    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_cpl_params_nlm{eb}.pt")
    nlm_model.diff_eqs = cpl.system
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_mean, nlm_std = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([x_test], n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals([x_test])

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_cpl_params_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = cpl.system
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_cpl_params_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x_test)], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp[0])
        hmc_residuals.append(hmc_residuals_tmp[0])
    hmc_samples = np.concatenate(hmc_samples, axis=0)
    hmc_residuals = np.concatenate(hmc_residuals, axis=0)

    data = {
         "x": x_test,
         "analytic": analytic,
         "FCNN": solution,
         "BBB": (bbb_samples.mean(axis=0), bbb_samples.std(axis=0)),
         "NLM": (nlm_mean[0].detach().cpu(), nlm_std[0].detach().cpu()),
         "HMC": (hmc_samples.mean(axis=0), hmc_samples.std(axis=0)),
         "BBB_samples": [bbb_samples],
         "NLM_samples": nlm_samples,
         "HMC_samples": [hmc_samples],
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": [bbb_residuals],
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": [hmc_residuals],
         }
    return data

@dill_dec_old("plot_data/cpl_params_bundle.dill", "plot_data/cpl_params_eb_bundle.dill")
def get_bundle_plot_data(eb=False):
    eb = "_eb" if eb else ""

    t_test = np.linspace(cpl.coords_test_min, cpl.coords_test_max, cpl.bundle_plot_dimension_sizes[0])
    params_test = [np.linspace(cpl.bundle_parameters_min_plot[i], cpl.bundle_parameters_max_plot[i], cpl.bundle_plot_dimension_sizes[i+1]) for i in range(len(cpl.bundle_parameters_min))]
    x_test = [x.reshape(-1, 1) for x in np.meshgrid(t_test, *params_test, indexing="ij")]
    
    analytic = analytic_cpl_bundle(*x_test)

    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = BundleSolver1D.load("checkpoints/solver_bundle_cpl_params_fcnn.ndeq")
    solver.conditions = cpl.bundle_conditions
    solution = solver.get_solution()(*x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(*x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_bundle_cpl_params_bbb{eb}.pyro", map_location="cpu")
    solver_bbb.diff_eqs = cpl.system_bundle
    solver_bbb.conditions = cpl.bundle_conditions
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x) for x in x_test], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp[0])
        bbb_residuals.append(bbb_residuals_tmp[0])
    bbb_samples = np.concatenate(bbb_samples, axis=0)
    bbb_residuals = np.concatenate(bbb_residuals, axis=0)

    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_bundle_cpl_params_nlm{eb}.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_model.diff_eqs = cpl.system_bundle
    nlm_mean, nlm_std = nlm_model.posterior_predictive(x_test, include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive(x_test, n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals(x_test)

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_bundle_cpl_params_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.conditions = cpl.bundle_conditions
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_solver.diff_eqs = cpl.system_bundle
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_cpl_params_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x) for x in x_test], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp[0])
        hmc_residuals.append(hmc_residuals_tmp[0])
    hmc_samples = np.concatenate(hmc_samples, axis=0)
    hmc_residuals = np.concatenate(hmc_residuals, axis=0)

    data = {
         "x": x_test,
         "analytic": analytic.reshape(cpl.bundle_plot_dimension_sizes),
         "FCNN": solution.reshape(cpl.bundle_plot_dimension_sizes),
         "BBB": (bbb_samples.mean(axis=0).reshape(cpl.bundle_plot_dimension_sizes), bbb_samples.std(axis=0).reshape(cpl.bundle_plot_dimension_sizes)),
         "NLM": (nlm_mean[0].detach().cpu().reshape(cpl.bundle_plot_dimension_sizes), nlm_std[0].detach().cpu().reshape(cpl.bundle_plot_dimension_sizes)),
         "HMC": (hmc_samples.mean(axis=0).reshape(cpl.bundle_plot_dimension_sizes), hmc_samples.std(axis=0).reshape(cpl.bundle_plot_dimension_sizes)),
         "BBB_samples": [bbb_samples],
         "NLM_samples": nlm_samples,
         "HMC_samples": [hmc_samples],
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": [bbb_residuals],
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": [hmc_residuals],
         }
    return data

@dill_dec_old("plot_data/cpl_params_best_fit.dill", "plot_data/cpl_params_best_fit_eb.dill")
def get_best_fit_plot_data(eb=False):
    eb = "_eb" if eb else ""
    w_0_fcnn, w_1_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_cpl_params_fcnn_cc.npy").mean(axis=0)
    w_0_bbb, w_1_bbb, Om_m_0_bbb, H_0_bbb = np.load(f"checkpoints/inverse_samples_bundle_cpl_params_bbb{eb}_cc.npy").mean(axis=0)
    w_0_nlm, w_1_nlm, Om_m_0_nlm, H_0_nlm = np.load(f"checkpoints/inverse_samples_bundle_cpl_params_nlm{eb}_cc.npy").mean(axis=0)
    w_0_hmc, w_1_hmc, Om_m_0_hmc, H_0_hmc = np.load(f"checkpoints/inverse_samples_bundle_cpl_params_hmc{eb}_cc.npy").mean(axis=0)

    z = np.linspace(0, 2, 100).reshape(-1, 1)

    solver_fcnn = BundleSolver1D.load("checkpoints/solver_bundle_cpl_params_fcnn.ndeq")
    solver_fcnn.conditions = cpl.bundle_conditions
    solution_fcnn = solver_fcnn.get_solution()

    bbb_solver = torch.load(f"checkpoints/solver_bundle_cpl_params_bbb{eb}.pyro", map_location="cpu")
    bbb_solver.conditions = cpl.bundle_conditions
    if "get_likelihood_std" in bbb_solver.__dict__:
        bbb_solver.get_likelihood_std.device = "cpu"

    nlm_model = NLMModel.load(f"checkpoints/model_bundle_cpl_params_nlm{eb}.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"

    hmc_solver = torch.load(f"checkpoints/solver_bundle_cpl_params_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.conditions = cpl.bundle_conditions
    hmc_solver.device = "cpu"
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_cpl_params_hmc{eb}.pyro", pickle_module=dill)

    bbb_sampler = bbb_sample_solutions(bbb_solver, 10_000, 2)
    nlm_sampler = nlm_sample_solutions(nlm_model, 10_000, 2)
    hmc_sampler = hmc_sample_solutions(hmc_solver, hmc_posterior_samples, 10_000, 2)

    with torch.no_grad():
        hubble_fcnn = cpl.H_CPL(z, w_0_fcnn, w_1_fcnn, Om_m_0_fcnn, H_0_fcnn, solution_fcnn)
        hubble_bbb = cpl.H_CPL(z, w_0_bbb, w_1_bbb, Om_m_0_bbb, H_0_bbb, bbb_sampler)
        hubble_nlm = cpl.H_CPL(z, w_0_nlm, w_1_nlm, Om_m_0_nlm, H_0_nlm, nlm_sampler)
        hubble_hmc = cpl.H_CPL(z, w_0_hmc, w_1_hmc, Om_m_0_hmc, H_0_hmc, hmc_sampler)

    bbb_mean, bbb_std = np.nanmean(hubble_bbb, axis=0), np.nanstd(hubble_bbb, axis=0)
    nlm_mean, nlm_std = np.nanmean(hubble_nlm, axis=0), np.nanstd(hubble_nlm, axis=0)
    hmc_mean, hmc_std = np.nanmean(hubble_hmc, axis=0), np.nanstd(hubble_hmc, axis=0)

    return {
        "x": z,
        "FCNN": hubble_fcnn.ravel(),
        "BBB": (bbb_mean.ravel(), bbb_std.ravel()),
        "NLM": (nlm_mean.ravel(), nlm_std.ravel()),
        "HMC": (hmc_mean.ravel(), hmc_std.ravel())
    }

def plot_cpl(eb=False, force=False):
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(cpl.coords_train_min[0], cpl.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
        axes[i].plot(data["x"], data["analytic"], 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Ground Truth')
        axes[i].plot(data["x"], data["FCNN"], "red", alpha=.6, label="Det Solution")
        axes[i].plot(data["x"], data[method][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-1*data[method][1].ravel(), data[method][0].ravel()+1*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-2*data[method][1].ravel(), data[method][0].ravel()+2*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-3*data[method][1].ravel(), data[method][0].ravel()+3*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

        axes[i].set_xlim(data["x"].min(), data["x"].max())
        axes[i].set_title(method, size=26)
        axes[i].set_xlabel("$z$", size=26)
        if i == 0: axes[i].set_ylabel("$x_{DE}(z)$", size=26)
        axes[i].set_ylim(0.6, 1.2)

    fig.suptitle("CPL Forward Solutions" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))
    plt.savefig(f"figures/cpl_params/forward_cpl_params{eb_text}.png", bbox_inches='tight')

def plot_cpl_bundle_errors(method, eb=False, force=False):
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb, force=force)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    w1_min, w1_max = data["x"][2].min(), data["x"][2].max()

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(data["FCNN"], None, data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)
    nlm_errors = error_fn(data["NLM"][0], data["NLM"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)
    bbb_errors = error_fn(data["BBB"][0], data["BBB"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(data["HMC"][0], data["HMC"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)

    for w0 in np.linspace(0, 3, 4):
        w0 = int(w0)

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))
        im1 = ax[0].imshow(fcnn_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Solution " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Test Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        cb1 = fig.colorbar(im1, ax=ax[0])
        cb2 = fig.colorbar(im2, ax=ax[1])
        cb3 = fig.colorbar(im3, ax=ax[2])
        cb3 = fig.colorbar(im4, ax=ax[3])
        fig.savefig(f"figures/cpl_params/bundle_error_cpl_params{eb_text}_{method}_test_{w0}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))

        im5 = ax[0].imshow(fcnn_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im6 = ax[1].imshow(nlm_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im7 = ax[2].imshow(bbb_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im8 = ax[3].imshow(hmc_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Solution " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Train Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        fig.colorbar(im5, ax=ax[0])
        fig.colorbar(im6, ax=ax[1])
        fig.colorbar(im7, ax=ax[2])
        fig.colorbar(im8, ax=ax[3])
        fig.savefig(f"figures/cpl_params/bundle_error_cpl_params{eb_text}_{method}_train_{w0}.png", bbox_inches='tight')

def plot_cpl_bundle_std_errors(method, eb=False, force=False):
    eb_text = "_eb" if eb else ""
    method = "std_" + method
    data = get_bundle_plot_data(eb=eb, force=force)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    w1_min, w1_max = data["x"][2].min(), data["x"][2].max()

    error_fn = error_metrics[method]
    nlm_errors = error_fn(data["NLM"][0], data["NLM"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)
    bbb_errors = error_fn(data["BBB"][0], data["BBB"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(data["HMC"][0], data["HMC"][1], data["analytic"]).reshape(cpl.bundle_plot_dimension_sizes)

    for w0 in np.linspace(0, 3, 4):
        w0 = int(w0)

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        im2 = ax[0].imshow(nlm_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Solution " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Test Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        cb2 = fig.colorbar(im2, ax=ax[0])
        cb3 = fig.colorbar(im3, ax=ax[1])
        cb3 = fig.colorbar(im4, ax=ax[2])
        fig.savefig(f"figures/cpl_params/bundle_std_error_cpl_params{eb_text}_{method}_test_{w0}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))

        im6 = ax[0].imshow(nlm_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im7 = ax[1].imshow(bbb_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im8 = ax[2].imshow(hmc_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Solution " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Train Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        fig.colorbar(im6, ax=ax[0])
        fig.colorbar(im7, ax=ax[1])
        fig.colorbar(im8, ax=ax[2])
        fig.savefig(f"figures/cpl_params/bundle_std_error_cpl_params{eb_text}_{method}_train_{w0}.png", bbox_inches='tight')

def plot_bundle_examples(eb=False):
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb)
    x = data["x"][0].reshape(cpl.bundle_plot_dimension_sizes)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()

    fig, ax = plt.subplots(1, 4, figsize=(24, 4))
    for w0 in map(int, np.linspace(0, cpl.bundle_plot_dimension_sizes[1]-1, 5)):
        fig, ax = plt.subplots(1, 4, figsize=(24, 4))
        for w1_idx in map(int, np.linspace(0, cpl.bundle_plot_dimension_sizes[2]-30, 5)):
            im1 = ax[0].plot(x[:, 0, 0], data["FCNN"].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], label="$w_1=" + f"{round(data['x'][2].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, w1_idx], 2)}$")
            im2 = ax[1].plot(x[:, 0, 0], data["NLM"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], label="$w_1=" + f"{round(data['x'][2].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, w1_idx], 2)}$")
            im3 = ax[2].plot(x[:, 0, 0], data["BBB"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], label="$w_1=" + f"{round(data['x'][2].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, w1_idx], 2)}$")
            im4 = ax[3].plot(x[:, 0, 0], data["HMC"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], label="$w_1=" + f"{round(data['x'][2].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, w1_idx], 2)}$")

            ax[1].fill_between(x[:, 0, 0], data["NLM"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]-2*data["NLM"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], data["NLM"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]+2*data["NLM"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[2].fill_between(x[:, 0, 0], data["BBB"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]-2*data["BBB"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], data["BBB"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]+2*data["BBB"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[3].fill_between(x[:, 0, 0], data["HMC"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]-2*data["HMC"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], data["HMC"][0].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx]+2*data["HMC"][1].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            
            ax[0].plot(x[:, 0, 0], data["analytic"].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], "--", color=im1[0].get_color())
            ax[1].plot(x[:, 0, 0], data["analytic"].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], "--", color=im1[0].get_color())
            ax[2].plot(x[:, 0, 0], data["analytic"].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], "--", color=im1[0].get_color())
            ax[3].plot(x[:, 0, 0], data["analytic"].reshape(cpl.bundle_plot_dimension_sizes)[:, w0, w1_idx], "--", color=im1[0].get_color())

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$x_{DE}$", size=20)
            ax[i].set_title(m, size=20)
            ax[i].axvspan(x[:, 0, 0][0], x[:, 0, 0][int(cpl.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')
            ax[i].set_xlim(z_min, z_max)

    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("CPL Bundle Solutions" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    plt.savefig(f"figures/cpl_params/bundle_examples_cpl_params{eb_text}.png", bbox_inches='tight')

def plot_hubble_forward(eb=False, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force)

    hubble_an = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["analytic"])
    hubble_fcnn = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["FCNN"])
    hubble_bbb = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["BBB_samples"][0])
    hubble_nlm = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["NLM_samples"][0])
    hubble_hmc = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["HMC_samples"][0])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(cpl.coords_train_min[0], cpl.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
        axes[i].plot(data["x"], hubble_an, 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Ground Truth')
        axes[i].plot(data["x"], hubble_fcnn, "red", alpha=.6, label="Det Solution")
        axes[i].plot(data["x"], stats[method][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-1*stats[method][1].ravel(), stats[method][0].ravel()+1*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-2*stats[method][1].ravel(), stats[method][0].ravel()+2*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-3*stats[method][1].ravel(), stats[method][0].ravel()+3*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

        axes[i].set_xlim(data["x"].min(), data["x"].max())
        axes[i].set_title(method, size=26)
        axes[i].set_xlabel("$z$", size=26)
        if i == 0: axes[i].set_ylabel("$H(z)$", size=26)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))
    plt.suptitle("CPL Hubble Forward"  + (" (Error Bounds)" if eb else ""), y=1.05, size=26)
    plt.savefig(f"figures/cpl_params/hubble_cpl_params{eb_text}{nan_text}.png", bbox_inches='tight')

def plot_hubble_best_fit(eb=False, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force)

    w_0_fcnn, w_1_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_cpl_params_fcnn_cc.npy").mean(axis=0)
    w_0_bbb, w_1_bbb, Om_m_0_bbb, H_0_bbb = np.load("checkpoints/inverse_samples_bundle_cpl_params_bbb_cc.npy").mean(axis=0)
    w_0_nlm, w_1_nlm, Om_m_0_nlm, H_0_nlm = np.load("checkpoints/inverse_samples_bundle_cpl_params_nlm_cc.npy").mean(axis=0)
    w_0_hmc, w_1_hmc, Om_m_0_hmc, H_0_hmc = np.load("checkpoints/inverse_samples_bundle_cpl_params_hmc_cc.npy").mean(axis=0)

    hubble_fcnn = cpl.H_CPL(data["x"], w_0_fcnn, w_1_fcnn, Om_m_0_fcnn, H_0_fcnn, data["FCNN"])
    hubble_bbb = cpl.H_CPL(data["x"], w_0_bbb, w_1_bbb, Om_m_0_bbb, H_0_bbb, data["BBB_samples"][0])
    hubble_nlm = cpl.H_CPL(data["x"], w_0_nlm, w_1_nlm, Om_m_0_nlm, H_0_nlm, data["NLM_samples"][0])
    hubble_hmc = cpl.H_CPL(data["x"], w_0_hmc, w_1_hmc, Om_m_0_hmc, H_0_hmc, data["HMC_samples"][0])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(cpl.coords_train_min[0], cpl.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
        axes[i].plot(data["x"], hubble_fcnn, "red", alpha=.6, label="Det Solution")
        axes[i].plot(data["x"], stats[method][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-1*stats[method][1].ravel(), stats[method][0].ravel()+1*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-2*stats[method][1].ravel(), stats[method][0].ravel()+2*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
        axes[i].fill_between(data["x"].ravel(), stats[method][0].ravel()-3*stats[method][1].ravel(), stats[method][0].ravel()+3*stats[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

        axes[i].set_xlim(data["x"].min(), data["x"].max())
        axes[i].set_title(method, size=26)
        axes[i].set_xlabel("$z$", size=26)
        if i == 0: axes[i].set_ylabel("$H(z)$", size=26)
        #axes[i].set_ylim(-1, 72)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))
    plt.suptitle("CPL Hubble Best Fit"  + (" (Error Bounds)" if eb else ""), y=1.05, size=26)
    plt.savefig(f"figures/cpl_params/hubble_cpl_best_fit{eb_text}{nan_text}.png", bbox_inches='tight')

def plot_hubble_bundle_errors(method, eb=False, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb, force=force)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    w1_min, w1_max = data["x"][2].min(), data["x"][2].max()

    z, w_0, w_1 = data["x"][0].reshape(cpl.bundle_plot_dimension_sizes), data["x"][1].reshape(cpl.bundle_plot_dimension_sizes), data["x"][2].reshape(cpl.bundle_plot_dimension_sizes)

    hubble_an = cpl.H_CPL(z, w_0, w_1, .3, 65, data["analytic"])
    hubble_fcnn = cpl.H_CPL(z, w_0, w_1, .3, 65, data["FCNN"])
    hubble_bbb = cpl.H_CPL(z, w_0, w_1, .3, 65, data["BBB_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_nlm = cpl.H_CPL(z, w_0, w_1, .3, 65, data["NLM_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_hmc = cpl.H_CPL(z, w_0, w_1, .3, 65, data["HMC_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))

    hubble_bbb_mean = np.__dict__[mean_fn](hubble_bbb, axis=0)
    hubble_nlm_mean = np.__dict__[mean_fn](hubble_nlm, axis=0)
    hubble_hmc_mean = np.__dict__[mean_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(hubble_fcnn, None, hubble_an)
    bbb_errors = error_fn(hubble_bbb_mean, None, hubble_an)
    nlm_errors = error_fn(hubble_nlm_mean, None, hubble_an)
    hmc_errors = error_fn(hubble_hmc_mean, None, hubble_an)

    for w0 in np.linspace(0, 3, 4):
        w0 = int(w0)

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))
        im1 = ax[0].imshow(fcnn_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Hubble " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Test Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        cb1 = fig.colorbar(im1, ax=ax[0])
        cb2 = fig.colorbar(im2, ax=ax[1])
        cb3 = fig.colorbar(im3, ax=ax[2])
        cb3 = fig.colorbar(im4, ax=ax[3])
        fig.savefig(f"figures/cpl_params/hubble_bundle_error_cpl_params{eb_text}{nan_text}_{method}_test_{w0}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))

        im5 = ax[0].imshow(fcnn_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im6 = ax[1].imshow(nlm_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im7 = ax[2].imshow(bbb_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im8 = ax[3].imshow(hmc_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Hubble " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Train Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        fig.colorbar(im5, ax=ax[0])
        fig.colorbar(im6, ax=ax[1])
        fig.colorbar(im7, ax=ax[2])
        fig.colorbar(im8, ax=ax[3])
        fig.savefig(f"figures/cpl_params/hubble_bundle_error_cpl_params{eb_text}{nan_text}_{method}_train_{w0}.png", bbox_inches='tight')

def plot_hubble_bundle_std_errors(method, eb=False, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    method = "std_" + method
    data = get_bundle_plot_data(eb=eb, force=force)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    w1_min, w1_max = data["x"][2].min(), data["x"][2].max()

    z, w_0, w_1 = data["x"][0].reshape(cpl.bundle_plot_dimension_sizes), data["x"][1].reshape(cpl.bundle_plot_dimension_sizes), data["x"][2].reshape(cpl.bundle_plot_dimension_sizes)

    hubble_an = cpl.H_CPL(z, w_0, w_1, .3, 65, data["analytic"])
    hubble_bbb = cpl.H_CPL(z, w_0, w_1, .3, 65, data["BBB_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_nlm = cpl.H_CPL(z, w_0, w_1, .3, 65, data["NLM_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_hmc = cpl.H_CPL(z, w_0, w_1, .3, 65, data["HMC_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))

    hubble_bbb_mean, hubble_bbb_std = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    hubble_nlm_mean, hubble_nlm_std = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    hubble_hmc_mean, hubble_hmc_std = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    bbb_errors = error_fn(hubble_bbb_mean, hubble_bbb_std, hubble_an).reshape(cpl.bundle_plot_dimension_sizes)
    nlm_errors = error_fn(hubble_nlm_mean, hubble_nlm_std, hubble_an).reshape(cpl.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(hubble_hmc_mean, hubble_hmc_std, hubble_an).reshape(cpl.bundle_plot_dimension_sizes)

    for w0 in np.linspace(0, 3, 4):
        w0 = int(w0)

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        im2 = ax[0].imshow(nlm_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Hubble " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Test Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        cb2 = fig.colorbar(im2, ax=ax[0])
        cb3 = fig.colorbar(im3, ax=ax[1])
        cb3 = fig.colorbar(im4, ax=ax[2])
        fig.savefig(f"figures/cpl_params/hubble_bundle_std_error_cpl_params{eb_text}{nan_text}_{method}_test_{w0}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))

        im6 = ax[0].imshow(nlm_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im7 = ax[1].imshow(bbb_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")
        im8 = ax[2].imshow(hmc_errors[:int(cpl.bundle_plot_dimension_sizes[0]/2), w0, :].T, cmap="viridis", origin="lower", extent=[z_min, z_max/2, w1_min, w1_max], aspect="auto")

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$w_1$", size=20)
            ax[i].set_title(m, size=20)
        fig.suptitle("CPL Hubble " + error_names[method] + " for " + f"$w_0 = {round(data['x'][1].reshape(cpl.bundle_plot_dimension_sizes)[0, w0, 0], 2)}$ (Train Region)" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

        fig.colorbar(im6, ax=ax[0])
        fig.colorbar(im7, ax=ax[1])
        fig.colorbar(im8, ax=ax[2])
        fig.savefig(f"figures/cpl_params/hubble_bundle_std_error_cpl_params{eb_text}{nan_text}_{method}_train_{w0}.png", bbox_inches='tight')

def plot_calibration(data, name):
    eb = "eb" in name
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    uct.plot_calibration(data["BBB"][0].ravel(), data["BBB"][1].ravel(), data["analytic"].ravel(), ax=axes[0])
    uct.plot_calibration(data["NLM"][0].numpy().ravel(), data["NLM"][1].numpy().ravel(), data["analytic"].ravel(), ax=axes[1])
    uct.plot_calibration(data["HMC"][0].ravel(), data["HMC"][1].ravel(), data["analytic"].ravel(), ax=axes[2])
    axes[0].set_title("BBB", size=26)
    axes[1].set_title("NLM", size=26)
    axes[2].set_title("HMC", size=26)

    fig.suptitle("CPL Forward Solution Calibration" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    plt.savefig(f"figures/cpl_params/{name}.png", bbox_inches='tight')

if __name__ == "__main__":
    plot_cpl()
    plot_cpl(eb=True)
    plot_hubble_forward()
    plot_hubble_forward(ignore_nans=True)
    plot_hubble_forward(eb=True)
    plot_hubble_forward(eb=True, ignore_nans=True)
    plot_hubble_best_fit()
    plot_hubble_best_fit(ignore_nans=True)
    plot_hubble_best_fit(eb=True)
    plot_hubble_best_fit(eb=True, ignore_nans=True)
    plot_cpl_bundle_errors(method="ae")
    plot_cpl_bundle_errors(method="re")
    plot_cpl_bundle_errors(method="rpd")
    plot_cpl_bundle_errors(method="ae", eb=True)
    plot_cpl_bundle_errors(method="re", eb=True)
    plot_cpl_bundle_errors(method="rpd", eb=True)
    plot_cpl_bundle_std_errors(method="ae")
    plot_cpl_bundle_std_errors(method="re")
    plot_cpl_bundle_std_errors(method="rpd")
    plot_cpl_bundle_std_errors(method="ae", eb=True)
    plot_cpl_bundle_std_errors(method="re", eb=True)
    plot_cpl_bundle_std_errors(method="rpd", eb=True)
    plot_bundle_examples()
    plot_bundle_examples(eb=True)
    plot_hubble_bundle_errors(method="ae")
    plot_hubble_bundle_errors(method="re")
    plot_hubble_bundle_errors(method="rpd")
    plot_hubble_bundle_errors(method="ae", ignore_nans=True)
    plot_hubble_bundle_errors(method="re", ignore_nans=True)
    plot_hubble_bundle_errors(method="rpd", ignore_nans=True)
    plot_hubble_bundle_errors(method="ae", eb=True)
    plot_hubble_bundle_errors(method="re", eb=True)
    plot_hubble_bundle_errors(method="rpd", eb=True)
    plot_hubble_bundle_errors(method="ae", eb=True, ignore_nans=True)
    plot_hubble_bundle_errors(method="re", eb=True, ignore_nans=True)
    plot_hubble_bundle_errors(method="rpd", eb=True, ignore_nans=True)
    plot_hubble_bundle_std_errors(method="ae")
    plot_hubble_bundle_std_errors(method="re")
    plot_hubble_bundle_std_errors(method="rpd")
    plot_hubble_bundle_std_errors(method="ae", ignore_nans=True)
    plot_hubble_bundle_std_errors(method="re", ignore_nans=True)
    plot_hubble_bundle_std_errors(method="rpd", ignore_nans=True)
    plot_hubble_bundle_std_errors(method="ae", eb=True)
    plot_hubble_bundle_std_errors(method="re", eb=True)
    plot_hubble_bundle_std_errors(method="rpd", eb=True)
    plot_hubble_bundle_std_errors(method="ae", eb=True, ignore_nans=True)
    plot_hubble_bundle_std_errors(method="re", eb=True, ignore_nans=True)
    plot_hubble_bundle_std_errors(method="rpd", eb=True, ignore_nans=True)
    plot_calibration(get_plot_data(), "calibration_cpl_params_forward")
    plot_calibration(get_bundle_plot_data(), "calibration_cpl_params_bundle")
    plot_calibration(get_plot_data(eb=True), "calibration_cpl_params_forward_eb")
    plot_calibration(get_bundle_plot_data(eb=True), "calibration_cpl_params_bundle_eb")
