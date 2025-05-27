from gc import collect
import numpy as np
from tqdm.auto import tqdm
import dill
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from neurodiffeq.solvers import Solver1D, BundleSolver1D
import torch
import equations.lcdm as lcdm
from models.nlm import NLMModel
from plotters.common import plot_calibration_area
from plotters.datasets import load_cc
from plotters.utils import dill_dec, dill_dec_old
from inverse.bayesian_adapters import nlm_sample_solutions, bbb_sample_solutions, hmc_sample_solutions
from .config import *
from .utils import error_metrics

error_names = {
    "re": "Relative Error",
    "ae": "Absolute Error",
    "rpd": "Relative Percent Difference",
    "std_re": "Std. Relative Error",
    "std_ae": "Std. Absolute Error",
    "std_rpd": "Std. Relative Percent Difference",
}

def analytic_lcdm(z, Om_m_0):
    c = (Om_m_0*3)**(1/3)
    return ((1 + z)*c)**3/3

@dill_dec("lcdm")
def get_plot_data(eb=False, domain_type="test"):
    eb = "_eb" if eb else ""

    if domain_type == "test":
        x_test = np.linspace(lcdm.coords_test_min, lcdm.coords_test_max, 200).reshape(-1, 1)
    elif domain_type == "train":
        x_test = np.linspace(lcdm.coords_train_min, lcdm.coords_train_max, 200).reshape(-1, 1)
    else:
        x_test = np.linspace(lcdm.coords_train_max, lcdm.coords_test_max, 200).reshape(-1, 1)

    analytic = analytic_lcdm(x_test, lcdm.Om_m_0)

    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = Solver1D.load("checkpoints/solver_lcdm_fcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_lcdm_bbb{eb}.pyro", map_location="cpu")
    solver_bbb.diff_eqs = lcdm.system
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
    nlm_model = NLMModel.load(f"checkpoints/model_lcdm_nlm{eb}.pt")
    nlm_model.diff_eqs = lcdm.system
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_mean, nlm_std = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([x_test], n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals([x_test])

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_lcdm_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = lcdm.system
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_lcdm_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
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
         "domain_type": domain_type,
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

@dill_dec("lcdm", bundle=True)
def get_bundle_plot_data(eb=False, domain_type="test"):
    eb = "_eb" if eb else ""

    if domain_type == "test":
        t_test = np.linspace(lcdm.coords_test_min, lcdm.coords_test_max, lcdm.bundle_plot_dimension_sizes[0])
    elif domain_type == "train":
        t_test = np.linspace(lcdm.coords_train_min, lcdm.coords_train_max, lcdm.bundle_plot_dimension_sizes[0])
    else:
        t_test = np.linspace(lcdm.coords_train_max, lcdm.coords_test_max, lcdm.bundle_plot_dimension_sizes[0])

    params_test = [np.linspace(lcdm.bundle_parameters_min_plot[i], lcdm.bundle_parameters_max_plot[i], lcdm.bundle_plot_dimension_sizes[i+1]) for i in range(len(lcdm.bundle_parameters_min))]
    x_test = [x.reshape(-1, 1) for x in np.meshgrid(t_test, *params_test, indexing="ij")]

    analytic = analytic_lcdm(*x_test)

    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = BundleSolver1D.load("checkpoints/solver_bundle_lcdm_fcnn.ndeq")
    solution = solver.get_solution()(*x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(*x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_bundle_lcdm_bbb{eb}.pyro", map_location="cpu")
    solver_bbb.diff_eqs = lcdm.system_bundle
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
    nlm_model = NLMModel.load(f"checkpoints/model_bundle_lcdm_nlm{eb}.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_model.diff_eqs = lcdm.system_bundle
    nlm_mean, nlm_std = nlm_model.posterior_predictive(x_test, include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive(x_test, n_samples=100, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals(x_test)

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_bundle_lcdm_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = lcdm.system_bundle
    hmc_solver.device = "cpu"
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_lcdm_hmc{eb}.pyro", pickle_module=dill)
    
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
        "domain_type": domain_type,
        "x": x_test,
        "analytic": analytic.reshape(lcdm.bundle_plot_dimension_sizes),
        "FCNN": solution.reshape(lcdm.bundle_plot_dimension_sizes),
        "BBB": (bbb_samples.mean(axis=0).reshape(lcdm.bundle_plot_dimension_sizes), bbb_samples.std(axis=0).reshape(lcdm.bundle_plot_dimension_sizes)),
        "NLM": (nlm_mean[0].detach().cpu().reshape(lcdm.bundle_plot_dimension_sizes), nlm_std[0].detach().cpu().reshape(lcdm.bundle_plot_dimension_sizes)),
        "HMC": (hmc_samples.mean(axis=0).reshape(lcdm.bundle_plot_dimension_sizes), hmc_samples.std(axis=0).reshape(lcdm.bundle_plot_dimension_sizes)),
        "BBB_samples": [bbb_samples],
        "NLM_samples": nlm_samples,
        "HMC_samples": [hmc_samples],
        "FCNN_residuals": fcnn_residuals,
        "BBB_residuals": [bbb_residuals],
        "NLM_residuals": [r.detach() for r in nlm_residuals],
        "HMC_residuals": [hmc_residuals],
    }
    return data

@dill_dec_old("plot_data/lcdm_best_fit.dill", "plot_data/lcdm_best_fit_eb.dill")
def get_best_fit_plot_data(eb=False):
    eb = "_eb" if eb else ""
    Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_lcdm_fcnn_cc.npy").mean(axis=0)
    Om_m_0_bbb, H_0_bbb = np.load(f"checkpoints/inverse_samples_bundle_lcdm_bbb{eb}_cc.npy").mean(axis=0)
    Om_m_0_nlm, H_0_nlm = np.load(f"checkpoints/inverse_samples_bundle_lcdm_nlm{eb}_cc.npy").mean(axis=0)
    Om_m_0_hmc, H_0_hmc = np.load(f"checkpoints/inverse_samples_bundle_lcdm_hmc{eb}_cc.npy").mean(axis=0)

    z = np.linspace(0, 2, 100).reshape(-1, 1)

    solver_fcnn = BundleSolver1D.load("checkpoints/solver_bundle_lcdm_fcnn.ndeq")
    solution_fcnn = solver_fcnn.get_solution()

    bbb_solver = torch.load(f"checkpoints/solver_bundle_lcdm_bbb{eb}.pyro", map_location="cpu")
    if "get_likelihood_std" in bbb_solver.__dict__:
        bbb_solver.get_likelihood_std.device = "cpu"

    nlm_model = NLMModel.load(f"checkpoints/model_bundle_lcdm_nlm{eb}.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"

    hmc_solver = torch.load(f"checkpoints/solver_bundle_lcdm_hmc{eb}.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.device = "cpu"
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_lcdm_hmc{eb}.pyro", pickle_module=dill)

    bbb_sampler = bbb_sample_solutions(bbb_solver, 10_000, 1)
    nlm_sampler = nlm_sample_solutions(nlm_model, 10_000, 1)
    hmc_sampler = hmc_sample_solutions(hmc_solver, hmc_posterior_samples, 10_000, 1)

    with torch.no_grad():
        hubble_fcnn = lcdm.H_LCDM(z, Om_m_0_fcnn, H_0_fcnn, solution_fcnn)
        hubble_bbb = lcdm.H_LCDM(z, Om_m_0_bbb, H_0_bbb, bbb_sampler)
        hubble_nlm = lcdm.H_LCDM(z, Om_m_0_nlm, H_0_nlm, nlm_sampler)
        hubble_hmc = lcdm.H_LCDM(z, Om_m_0_hmc, H_0_hmc, hmc_sampler)

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

def plot_lcdm(eb=False, force=False, domain_type="test"):
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force, domain_type=domain_type)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(lcdm.coords_train_min[0], lcdm.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
        axes[i].plot(data["x"], data["analytic"], 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Ground Truth')
        axes[i].plot(data["x"], data["FCNN"], "red", alpha=.6, label="Det Solution")
        axes[i].plot(data["x"], data[method][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-1*data[method][1].ravel(), data[method][0].ravel()+1*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-2*data[method][1].ravel(), data[method][0].ravel()+2*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
        axes[i].fill_between(data["x"].ravel(), data[method][0].ravel()-3*data[method][1].ravel(), data[method][0].ravel()+3*data[method][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

        axes[i].set_xlim(data["x"].min(), data["x"].max())
        axes[i].set_title(method, size=26)
        axes[i].set_xlabel("$z$", size=26)
        if i == 0: axes[i].set_ylabel("$x_m(z)$", size=26)
        axes[i].set_ylim(-1, 72)

    fig.suptitle("$\Lambda$CDM Forward Solutions" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))
    plt.savefig(f"figures/lcdm/forward_lcdm{eb_text}_{domain_type}.png", bbox_inches='tight')

def plot_lcdm_bundle_errors(method, eb=False, force=False, domain_type="test"):
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb, force=force, domain_type=domain_type)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    Om_min, Om_max = data["x"][1].min(), data["x"][1].max()

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(data["FCNN"], None, data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)
    nlm_errors = error_fn(data["NLM"][0], data["NLM"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)
    bbb_errors = error_fn(data["BBB"][0], data["BBB"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(data["HMC"][0], data["HMC"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)

    fig, ax = plt.subplots(1, 4, figsize=(24, 4))
    im1 = ax[0].imshow(fcnn_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im2 = ax[1].imshow(nlm_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im3 = ax[2].imshow(bbb_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im4 = ax[3].imshow(hmc_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")

    for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
        ax[i].set_xlabel("$z$", size=20)
        ax[i].set_ylabel("$\Omega_{m,0}$", size=20)
        ax[i].set_title(m, size=20)
    fig.suptitle("$\Lambda$CDM Solution " + error_names[method] + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])
    fig.colorbar(im4, ax=ax[3])
    fig.savefig(f"figures/lcdm/bundle_error_lcdm{eb_text}_{method}_{domain_type}.png", bbox_inches='tight')

def plot_lcdm_bundle_std_errors(method, eb=False, force=False, domain_type="test"):
    eb_text = "_eb" if eb else ""
    method = "std_" + method
    data = get_bundle_plot_data(eb=eb, force=force, domain_type=domain_type)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    Om_min, Om_max = data["x"][1].min(), data["x"][1].max()

    error_fn = error_metrics[method]
    print(data["NLM"][0].shape, data["analytic"].shape)
    nlm_errors = error_fn(data["NLM"][0], data["NLM"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)
    bbb_errors = error_fn(data["BBB"][0], data["BBB"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(data["HMC"][0], data["HMC"][1], data["analytic"]).reshape(lcdm.bundle_plot_dimension_sizes)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    im2 = ax[0].imshow(nlm_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im3 = ax[1].imshow(bbb_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im4 = ax[2].imshow(hmc_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")

    for i, m in enumerate(["NLM", "BBB", "HMC"]):
        ax[i].set_xlabel("$z$", size=20)
        ax[i].set_ylabel("$\Omega_{m,0}$", size=20)
        ax[i].set_title(m, size=20)
    fig.suptitle("$\Lambda$CDM Solution " + error_names[method] + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

    fig.colorbar(im2, ax=ax[0])
    fig.colorbar(im3, ax=ax[1])
    fig.colorbar(im4, ax=ax[2])
    fig.savefig(f"figures/lcdm/bundle_std_error_lcdm{eb_text}_{method}_{domain_type}.png", bbox_inches='tight')

def plot_bundle_examples(eb=False, domain_type="test"):
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb, domain_type=domain_type)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()

    fig, ax = plt.subplots(1, 4, figsize=(24, 4))
    for i in map(int, np.linspace(0, lcdm.bundle_plot_dimension_sizes[1]-1, 5)):
        im1 = ax[0].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["FCNN"].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(lcdm.bundle_plot_dimension_sizes)[0, i], 2)}$")
        im2 = ax[1].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["NLM"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(lcdm.bundle_plot_dimension_sizes)[0, i], 2)}$")
        im3 = ax[2].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["BBB"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(lcdm.bundle_plot_dimension_sizes)[0, i], 2)}$")
        im4 = ax[3].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["HMC"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(lcdm.bundle_plot_dimension_sizes)[0, i], 2)}$")

        ax[1].fill_between(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["NLM"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]-2*data["NLM"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], data["NLM"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]+2*data["NLM"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
        ax[2].fill_between(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["BBB"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]-2*data["BBB"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], data["BBB"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]+2*data["BBB"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
        ax[3].fill_between(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["HMC"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]-2*data["HMC"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], data["HMC"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, i]+2*data["HMC"][1].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
        
        ax[0].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], "--", color=im1[0].get_color())
        ax[1].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], "--", color=im1[0].get_color())
        ax[2].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], "--", color=im1[0].get_color())
        ax[3].plot(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0], data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes)[:, i], "--", color=im1[0].get_color())

    for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
        ax[i].set_xlabel("$z$", size=20)
        ax[i].set_ylabel("$x$", size=20)
        ax[i].set_title(m, size=20)
        ax[i].axvspan(data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0][0], data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes)[:, 0][int(lcdm.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')
        ax[i].set_xlim(z_min, z_max)
        ax[i].set_ylim(*ax[0].get_ylim())

    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("$\Lambda$CDM Bundle Solutions" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    plt.savefig(f"figures/lcdm/bundle_examples_lcdm{eb_text}_{domain_type}.png", bbox_inches='tight')

def plot_hubble_forward(eb=False, force=False, ignore_nans=False, domain_type="test"):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force, domain_type=domain_type)

    hubble_an = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["analytic"])
    hubble_fcnn = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["FCNN"])
    hubble_bbb = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["BBB_samples"][0])
    hubble_nlm = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["NLM_samples"][0])
    hubble_hmc = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["HMC_samples"][0])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(lcdm.coords_train_min[0], lcdm.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
        #axes[i].set_ylim(-1, 72)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("$\Lambda$CDM Hubble Forward" + (" (Error Bounds)" if eb else ""), y=1.05)
    plt.savefig(f"figures/lcdm/hubble_lcdm{eb_text}{nan_text}.png", bbox_inches='tight')

def plot_hubble_best_fit(eb=False, force=False, ignore_nans=False, domain_type="test"):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_plot_data(eb=eb, force=force, domain_type=domain_type)
    cc_z, cc_h, cc_std = load_cc("datasets/cc.csv")

    Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_lcdm_fcnn_cc.npy").mean(axis=0)
    Om_m_0_bbb, H_0_bbb = np.load("checkpoints/inverse_samples_bundle_lcdm_bbb_cc.npy").mean(axis=0)
    Om_m_0_nlm, H_0_nlm = np.load("checkpoints/inverse_samples_bundle_lcdm_nlm_cc.npy").mean(axis=0)
    Om_m_0_hmc, H_0_hmc = np.load("checkpoints/inverse_samples_bundle_lcdm_hmc_cc.npy").mean(axis=0)

    hubble_fcnn = lcdm.H_LCDM(data["x"], Om_m_0_fcnn, H_0_fcnn, data["FCNN"])
    hubble_bbb = lcdm.H_LCDM(data["x"], Om_m_0_bbb, H_0_bbb, data["BBB_samples"][0])
    hubble_nlm = lcdm.H_LCDM(data["x"], Om_m_0_nlm, H_0_nlm, data["NLM_samples"][0])
    hubble_hmc = lcdm.H_LCDM(data["x"], Om_m_0_hmc, H_0_hmc, data["HMC_samples"][0])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)
    
    colors = ["tab:green", "tab:red", "tab:purple"]
    fig, axes = plt.subplots(1, 1)
    axes.plot(data["x"], hubble_fcnn, "tab:orange", label="FCNN")
    axes.errorbar(cc_z, cc_h, yerr=cc_std, fmt='o', markersize=3)
    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes.plot(data["x"], stats[method][0], colors[i], linewidth=2, alpha=1, label=method)
        # axes.fill_between(data["x"].ravel(), stats[method][0].ravel()-1*stats[method][1].ravel(), stats[method][0].ravel()+1*stats[method][1].ravel(), color=colors[i], alpha=0.2, edgecolor=None)
        # axes.fill_between(data["x"].ravel(), stats[method][0].ravel()-2*stats[method][1].ravel(), stats[method][0].ravel()+2*stats[method][1].ravel(), color=colors[i], alpha=0.2, edgecolor=None)
        # axes.fill_between(data["x"].ravel(), stats[method][0].ravel()-3*stats[method][1].ravel(), stats[method][0].ravel()+3*stats[method][1].ravel(), color=colors[i], alpha=0.2, edgecolor=None)

    axes.set_title("$\Lambda$CDM", size=26)
    axes.set_xlim(data["x"].min(), 2.1)
    axes.set_xlabel("$z$", size=26)
    axes.set_ylabel("$H(z)$", size=26)

    axes.legend(fontsize=14)
    fig.suptitle("$\Lambda$CDM Hubble Best Fit" + (" (Error Bounds)" if eb else ""), y=1.05)
    plt.savefig(f"figures/lcdm/hubble_lcdm_best_fit{eb_text}{nan_text}.png", bbox_inches='tight')

def plot_hubble_bundle_errors(method, eb=False, force=False, ignore_nans=False, domain_type="test"):
    mean_fn = "nanmean" if ignore_nans else "mean"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    data = get_bundle_plot_data(eb=eb, force=force, domain_type=domain_type)

    z, Om_m_0 = data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes), data["x"][1].reshape(lcdm.bundle_plot_dimension_sizes)

    hubble_an = lcdm.H_LCDM(z, Om_m_0, 65, data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes))
    hubble_fcnn = lcdm.H_LCDM(z, Om_m_0, 65, data["FCNN"].reshape(lcdm.bundle_plot_dimension_sizes))
    hubble_bbb = lcdm.H_LCDM(z, Om_m_0, 65, data["BBB_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_nlm = lcdm.H_LCDM(z, Om_m_0, 65, data["NLM_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_hmc = lcdm.H_LCDM(z, Om_m_0, 65, data["HMC_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    Om_min, Om_max = data["x"][1].min(), data["x"][1].max()

    hubble_bbb_mean = np.__dict__[mean_fn](hubble_bbb, axis=0)
    hubble_nlm_mean = np.__dict__[mean_fn](hubble_nlm, axis=0)
    hubble_hmc_mean = np.__dict__[mean_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(hubble_fcnn, None, hubble_an)
    bbb_errors = error_fn(hubble_bbb_mean, None, hubble_an)
    nlm_errors = error_fn(hubble_nlm_mean, None, hubble_an)
    hmc_errors = error_fn(hubble_hmc_mean, None, hubble_an)

    fig, ax = plt.subplots(1, 4, figsize=(24, 4))
    im1 = ax[0].imshow(fcnn_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im2 = ax[1].imshow(nlm_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im3 = ax[2].imshow(bbb_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im4 = ax[3].imshow(hmc_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")

    for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
        ax[i].set_xlabel("$z$", size=20)
        ax[i].set_ylabel("$\Omega_{m,0}$", size=20)
        ax[i].set_title(m, size=20)
    fig.suptitle("$\Lambda$CDM Hubble " + error_names[method] + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])
    fig.colorbar(im4, ax=ax[3])
    fig.savefig(f"figures/lcdm/hubble_bundle_error_lcdm{eb_text}{nan_text}_{method}_{domain_type}.png", bbox_inches='tight')

def plot_hubble_bundle_std_errors(method, eb=False, force=False, ignore_nans=False, domain_type="test"):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    eb_text = "_eb" if eb else ""
    method = "std_" + method
    data = get_bundle_plot_data(eb=eb, force=force, domain_type=domain_type)

    z, Om_m_0 = data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes), data["x"][1].reshape(lcdm.bundle_plot_dimension_sizes)

    hubble_an = lcdm.H_LCDM(z, Om_m_0, 65, data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes))
    hubble_bbb = lcdm.H_LCDM(z, Om_m_0, 65, data["BBB_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_nlm = lcdm.H_LCDM(z, Om_m_0, 65, data["NLM_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_hmc = lcdm.H_LCDM(z, Om_m_0, 65, data["HMC_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    Om_min, Om_max = data["x"][1].min(), data["x"][1].max()

    hubble_bbb_mean, hubble_bbb_std = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    hubble_nlm_mean, hubble_nlm_std = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    hubble_hmc_mean, hubble_hmc_std = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    bbb_errors = error_fn(hubble_bbb_mean, hubble_bbb_std, hubble_an)
    nlm_errors = error_fn(hubble_nlm_mean, hubble_nlm_std, hubble_an)
    hmc_errors = error_fn(hubble_hmc_mean, hubble_hmc_std, hubble_an)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    im2 = ax[0].imshow(nlm_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im3 = ax[1].imshow(bbb_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")
    im4 = ax[2].imshow(hmc_errors.reshape(lcdm.bundle_plot_dimension_sizes).T, cmap="viridis", origin="lower", extent=[z_min, z_max, Om_min, Om_max], aspect="auto")

    for i, m in enumerate(["NLM", "BBB", "HMC"]):
        ax[i].set_xlabel("$z$", size=20)
        ax[i].set_ylabel("$\Omega_{m,0}$", size=20)
        ax[i].set_title(m, size=20)
    fig.suptitle("$\Lambda$CDM Hubble " + error_names[method] + (" (Error Bounds)" if eb else ""), size=26, y=1.05)

    fig.colorbar(im2, ax=ax[0])
    fig.colorbar(im3, ax=ax[1])
    fig.colorbar(im4, ax=ax[2])
    fig.savefig(f"figures/lcdm/hubble_bundle_std_error_lcdm{eb_text}{nan_text}_{method}_{domain_type}.png", bbox_inches='tight')

def plot_calibration(data, name):
    eb = "eb" in name
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    uct.plot_calibration(data["BBB"][0].ravel(), data["BBB"][1].ravel(), data["analytic"].ravel(), ax=axes[0])
    uct.plot_calibration(data["NLM"][0].numpy().ravel(), data["NLM"][1].numpy().ravel(), data["analytic"].ravel(), ax=axes[1])
    uct.plot_calibration(data["HMC"][0].ravel(), data["HMC"][1].ravel(), data["analytic"].ravel(), ax=axes[2])
    axes[0].set_title("BBB", size=26)
    axes[1].set_title("NLM", size=26)
    axes[2].set_title("HMC", size=26)
    fig.suptitle("$\Lambda$CDM Forward Solution Calibration" + (" (Error Bounds)" if eb else ""), size=26, y=1.05)
    plt.savefig(f"figures/lcdm/{name}.png", bbox_inches='tight')

def plot_calibration_dts(data_train, data_test, data_ood, name):
    method = "Bundle" if "bundle" in name else "Forward"
    eb = "eb" in name
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ["Training Domain", "Testing Domain", "OOD Domain"]
    print("BBB")
    plot_calibration_area(
        [data["BBB"][0].ravel() for data in [data_train, data_test, data_ood]],
        [data["BBB"][1].ravel() for data in [data_train, data_test, data_ood]],
        [data["analytic"].ravel() for data in [data_train, data_test, data_ood]],
        ax=axes[0],
        labels=labels
    )
    print("NLM")
    plot_calibration_area(
        [data["NLM"][0].numpy().ravel() for data in [data_train, data_test, data_ood]],
        [data["NLM"][1].numpy().ravel() for data in [data_train, data_test, data_ood]],
        [data["analytic"].ravel() for data in [data_train, data_test, data_ood]],
        ax=axes[1],
        labels=labels
    )
    print("HMC")
    plot_calibration_area(
        [data["HMC"][0].ravel() for data in [data_train, data_test, data_ood]],
        [data["HMC"][1].ravel() for data in [data_train, data_test, data_ood]],
        [data["analytic"].ravel() for data in [data_train, data_test, data_ood]],
        ax=axes[2],
        labels=labels
    )
    axes[0].annotate("BBB", size=26, xy=(0.5, 1), xytext=(0, 21), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[1].annotate("NLM", size=26, xy=(0.5, 1), xytext=(0, 21), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[2].annotate("HMC", size=26, xy=(0.5, 1), xytext=(0, 21), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[0].annotate("$x$", xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - 5, 0), xycoords=axes[0].yaxis.label, textcoords='offset points', size=26, ha='right', va='center')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.22))

    fig.suptitle(f"$\Lambda$CDM {method} Solution Calibration" + (" (Error Bounds)" if eb else ""), size=26, y=1.1)
    plt.savefig(f"figures/lcdm/{name}.png", bbox_inches='tight')

def plot_residuals(eb=False, domain_type="test"):
    data = get_plot_data(eb=eb, domain_type=domain_type)

    bbb_res = np.stack(data["BBB_residuals"]).reshape(len(data["BBB_residuals"]), -1).mean(axis=0)
    hmc_res = np.stack(data["HMC_residuals"]).reshape(len(data["HMC_residuals"]), -1).mean(axis=0)

    plt.plot(data["x"].ravel(), data["FCNN_residuals"].ravel(), label="FCNN")
    plt.plot(data["x"].ravel(), bbb_res, label="BBB")
    plt.plot(data["x"].ravel(), hmc_res, label="HMC")
    plt.plot(data["x"].ravel(), data["NLM_residuals"][0].ravel(), label="NLM")
    plt.legend()
    plt.savefig("figures/lcdm/residuals_forward.png")

if __name__ == "__main__":
    # plot_residuals()
    # plot_lcdm()
    # plot_lcdm(eb=True)
    # plot_hubble_forward()
    # plot_hubble_forward(ignore_nans=True)
    # plot_hubble_forward(eb=True)
    # plot_hubble_forward(eb=True, ignore_nans=True)
    # plot_hubble_best_fit()
    # plot_hubble_best_fit(ignore_nans=True)
    # plot_hubble_best_fit(eb=True)
    # plot_hubble_best_fit(eb=True, ignore_nans=True)
    # plot_lcdm_bundle_errors(method="ae")
    # plot_lcdm_bundle_errors(method="re")
    # plot_lcdm_bundle_errors(method="rpd")
    # plot_lcdm_bundle_errors(method="ae", eb=True)
    # plot_lcdm_bundle_errors(method="re", eb=True)
    # plot_lcdm_bundle_errors(method="rpd", eb=True)
    # plot_lcdm_bundle_std_errors(method="ae")
    # plot_lcdm_bundle_std_errors(method="re")
    # plot_lcdm_bundle_std_errors(method="rpd")
    # plot_lcdm_bundle_std_errors(method="ae", eb=True)
    # plot_lcdm_bundle_std_errors(method="re", eb=True)
    # plot_lcdm_bundle_std_errors(method="rpd", eb=True)
    # plot_bundle_examples()
    # plot_bundle_examples(eb=True)
    # plot_hubble_bundle_errors(method="ae")
    # plot_hubble_bundle_errors(method="re")
    # plot_hubble_bundle_errors(method="rpd")
    # plot_hubble_bundle_errors(method="ae", ignore_nans=True)
    # plot_hubble_bundle_errors(method="re", ignore_nans=True)
    # plot_hubble_bundle_errors(method="rpd", ignore_nans=True)
    # plot_hubble_bundle_errors(method="ae", eb=True)
    # plot_hubble_bundle_errors(method="re", eb=True)
    # plot_hubble_bundle_errors(method="rpd", eb=True)
    # plot_hubble_bundle_errors(method="ae", eb=True, ignore_nans=True)
    # plot_hubble_bundle_errors(method="re", eb=True, ignore_nans=True)
    # plot_hubble_bundle_errors(method="rpd", eb=True, ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="ae")
    # plot_hubble_bundle_std_errors(method="re")
    # plot_hubble_bundle_std_errors(method="rpd")
    # plot_hubble_bundle_std_errors(method="ae", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="re", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="rpd", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="ae", eb=True)
    # plot_hubble_bundle_std_errors(method="re", eb=True)
    # plot_hubble_bundle_std_errors(method="rpd", eb=True)
    # plot_hubble_bundle_std_errors(method="ae", eb=True, ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="re", eb=True, ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="rpd", eb=True, ignore_nans=True)

    # for dt in ["test", "train", "ood"]:
    #     plot_calibration(get_plot_data(domain_type=dt), f"calibration_lcdm_forward_{dt}")
    #     plot_calibration(get_bundle_plot_data(domain_type=dt), f"calibration_lcdm_bundle_{dt}")
    #     plot_calibration(get_plot_data(eb=True, domain_type=dt), f"calibration_lcdm_forward_eb_{dt}")
    #     plot_calibration(get_bundle_plot_data(eb=True, domain_type=dt), f"calibration_lcdm_bundle_eb_{dt}")
    plot_calibration_dts(*[get_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_lcdm_forward_all")
    plot_calibration_dts(*[get_plot_data(eb=True, domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_lcdm_forward_eb_all")
    plot_calibration_dts(*[get_bundle_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_lcdm_bundle_all")
    plot_calibration_dts(*[get_bundle_plot_data(eb=True, domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_lcdm_bundle_eb_all")