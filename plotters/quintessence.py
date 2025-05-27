from gc import collect
import numpy as np
import dill
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from neurodiffeq.solvers import Solver1D, BundleSolver1D
import torch
from tqdm.auto import tqdm
import equations.quintessence as quintessence
from models.nlm import NLMModel
from plotters.common import plot_calibration_area
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

def numerical_quintessence(t, lam_prime=quintessence.lam_prime, Om_m_0=quintessence.Om_m_0):
    def func(N_prime, Y):
        x, y = Y
        return [
                quintessence.N_0_abs*(-3*x + quintessence.lam_max*(np.sqrt(6)/2)*lam_prime*(y ** 2) + (3/2)*x*(1 + (x**2) - (y**2))),
                quintessence.N_0_abs*(-quintessence.lam_max*(np.sqrt(6)/2)*lam_prime*(y * x) + (3/2)*y*(1 + (x**2) - (y**2)))
                ]
     
    initial_conditions = np.array([0, ((1 - Om_m_0)/(Om_m_0*(np.e**(-3*(quintessence.N_prime_0 - 1)*quintessence.N_0_abs)) + 1 - Om_m_0)) ** (1/2)])
    rk4_sol = RK45(func, t0=quintessence.coords_test_min[0], y0=initial_conditions, t_bound=quintessence.coords_test_max[0], max_step=0.001)

    t_values = [quintessence.coords_test_min[0]]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])

    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)
    x = np.interp(t, rk4_t, rk4_x_points)
    y = np.interp(t, rk4_t, rk4_y_points)
    return x, y

def numerical_quintessence_bundle(t, lam_prime, Om_m_0):
    sol_x = np.zeros((t.shape[0], lam_prime.shape[0], Om_m_0.shape[0]))
    sol_y = np.zeros((t.shape[0], lam_prime.shape[0], Om_m_0.shape[0]))
    for i, l in tqdm(enumerate(lam_prime.ravel()), desc="Bundle Numerical Lambda Prime", total=lam_prime.shape[0]):
        for j, o in tqdm(enumerate(Om_m_0.ravel()), leave=False, desc="Bundle Numerical Om_m_0", total=Om_m_0.shape[0]):
            x, y = numerical_quintessence(t, l, o)
            sol_x[:, i, j] = x.ravel()
            sol_y[:, i, j] = y.ravel()
    return sol_x, sol_y


@dill_dec("quintessence")
def get_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        x_test = np.linspace(quintessence.coords_test_min, quintessence.coords_test_max, 200).reshape(-1, 1)
    elif domain_type == "train":
        x_test = np.linspace(quintessence.coords_train_min, quintessence.coords_train_max, 200).reshape(-1, 1)
    else:
        x_test = np.linspace(quintessence.coords_train_max, quintessence.coords_test_max, 200).reshape(-1, 1)
    numerical = numerical_quintessence(x_test)

    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = Solver1D.load("checkpoints/solver_quintessence_fcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_quintessence_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = quintessence.system
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x_test)], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp)
        bbb_residuals.append(bbb_residuals_tmp)
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(2)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(2)]

    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_quintessence_nlm.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_model.diff_eqs = quintessence.system
    nlm_means, nlm_stds = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([x_test], n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals([x_test])

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_quintessence_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = quintessence.system
    hmc_posterior_samples = torch.load(f"checkpoints/samples_quintessence_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_samples = hmc_solver.posterior_predictive([torch.tensor(x_test)], hmc_posterior_samples, to_numpy=True)
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x_test)], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp)
        hmc_residuals.append(hmc_residuals_tmp)
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(2)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(2)]

    data = {
        "domain_type": domain_type,
         "x": x_test,
         "numerical": numerical,
         "FCNN": solution,
         "BBB": [(bbb_samples[i].mean(axis=0), bbb_samples[i].std(axis=0)) for i in range(2)],
         "NLM": [(nlm_means[i].detach().cpu(), nlm_stds[i].detach().cpu()) for i in range(2)],
         "HMC": [(hmc_samples[i].mean(axis=0), hmc_samples[i].std(axis=0)) for i in range(2)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": hmc_residuals,
         }
    return data

def get_bundle_plot_data_fcnn(x_test):
    print("Loading FCNN")
    solver = BundleSolver1D.load("checkpoints/solver_bundle_quintessence_fcnn.ndeq")
    solution = solver.get_solution()(*x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(*x_test, to_numpy=True)
    return solution, fcnn_residuals

def get_bundle_plot_data_bbb(x_test, batches, batch_size):
    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_bundle_quintessence_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = quintessence.system_bundle
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x) for x in x_test], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp)
        bbb_residuals.append(bbb_residuals_tmp)
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(2)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(2)]
    return bbb_samples, bbb_residuals

def get_bundle_plot_data_nlm(x_test):
    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_bundle_quintessence_nlm.pt")
    nlm_model.diff_eqs = quintessence.system_bundle
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_mean, nlm_std = nlm_model.posterior_predictive(x_test, include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive(x_test, n_samples=10000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals(x_test)
    nlm_mean = [m.detach().cpu() for m in nlm_mean]
    nlm_std = [m.detach().cpu() for m in nlm_std]
    nlm_residuals = [r.detach() for r in nlm_residuals]
    return nlm_mean, nlm_std, nlm_samples, nlm_residuals

def get_bundle_plot_data_hmc(x_test, batch_size):
    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_bundle_quintessence_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = quintessence.system_bundle
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_quintessence_hmc.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x) for x in x_test], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp)
        hmc_residuals.append(hmc_residuals_tmp)
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(2)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(2)]
    return hmc_samples, hmc_residuals

@dill_dec("quintessence", bundle=True)
def get_bundle_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        t_test = np.linspace(quintessence.coords_test_min, quintessence.coords_test_max, quintessence.bundle_plot_dimension_sizes[0])
    elif domain_type == "train":
        t_test = np.linspace(quintessence.coords_train_min, quintessence.coords_train_max, quintessence.bundle_plot_dimension_sizes[0])
    else:
        t_test = np.linspace(quintessence.coords_train_max, quintessence.coords_test_max, quintessence.bundle_plot_dimension_sizes[0])

    params_test = [np.linspace(quintessence.bundle_parameters_min_plot[i], quintessence.bundle_parameters_max_plot[i], quintessence.bundle_plot_dimension_sizes[i+1]) for i in range(len(quintessence.bundle_parameters_min))]
    x_test = [x.reshape(-1, 1) for x in np.meshgrid(t_test, *params_test, indexing="ij")]
    numerical = numerical_quintessence_bundle(t_test, *params_test)

    batch_size = 10
    batches = 10000 // batch_size

    solution, fcnn_residuals = get_bundle_plot_data_fcnn(x_test)
    collect()
    bbb_samples, bbb_residuals= get_bundle_plot_data_bbb(x_test, batches, batch_size)
    collect()
    nlm_mean, nlm_std, nlm_samples, nlm_residuals = get_bundle_plot_data_nlm(x_test)
    collect()
    hmc_samples, hmc_residuals = get_bundle_plot_data_hmc(x_test, batch_size)
    collect()

    data = {
        "domain_type": domain_type,
         "x": x_test,
         "numerical": [num.reshape(quintessence.bundle_plot_dimension_sizes) for num in numerical],
         "FCNN": [sol.reshape(quintessence.bundle_plot_dimension_sizes) for sol in solution],
         "BBB": [(bbb_samples[i].mean(axis=0).reshape(quintessence.bundle_plot_dimension_sizes), bbb_samples[i].std(axis=0).reshape(quintessence.bundle_plot_dimension_sizes)) for i in range(2)],
         "NLM": [(nlm_mean[i].reshape(quintessence.bundle_plot_dimension_sizes), nlm_std[i].reshape(quintessence.bundle_plot_dimension_sizes)) for i in range(2)],
         "HMC": [(hmc_samples[i].mean(axis=0).reshape(quintessence.bundle_plot_dimension_sizes), hmc_samples[i].std(axis=0).reshape(quintessence.bundle_plot_dimension_sizes)) for i in range(2)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": nlm_residuals,
         "HMC_residuals": hmc_residuals,
         }
    return data

@dill_dec_old("plot_data/quintessence_best_fit.dill", "plot_data/quintessence_best_fit_eb.dill")
def get_best_fit_plot_data():
    lambda_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_quintessence_fcnn_cc.npy").mean(axis=0)
    lambda_bbb, Om_m_0_bbb, H_0_bbb = np.load(f"checkpoints/inverse_samples_bundle_quintessence_bbb_cc.npy").mean(axis=0)
    lambda_nlm, Om_m_0_nlm, H_0_nlm = np.load(f"checkpoints/inverse_samples_bundle_quintessence_nlm_cc.npy").mean(axis=0)
    lambda_hmc, Om_m_0_hmc, H_0_hmc = np.load(f"checkpoints/inverse_samples_bundle_quintessence_hmc_cc.npy").mean(axis=0)

    z = np.linspace(0, 2, 100).reshape(-1, 1)

    solver_fcnn = BundleSolver1D.load("checkpoints/solver_bundle_quintessence_fcnn.ndeq")
    solution_fcnn = solver_fcnn.get_solution()

    bbb_solver = torch.load(f"checkpoints/solver_bundle_quintessence_bbb.pyro", map_location="cpu")
    if "get_likelihood_std" in bbb_solver.__dict__:
        bbb_solver.get_likelihood_std.device = "cpu"

    nlm_model = NLMModel.load(f"checkpoints/model_bundle_quintessence_nlm.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"

    hmc_solver = torch.load(f"checkpoints/solver_bundle_quintessence_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.device = "cpu"
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_quintessence_hmc.pyro", pickle_module=dill)

    bbb_sampler = bbb_sample_solutions(bbb_solver, 10_000, 2)
    nlm_sampler = nlm_sample_solutions(nlm_model, 10_000, 2)
    hmc_sampler = hmc_sample_solutions(hmc_solver, hmc_posterior_samples, 10_000, 2)

    with torch.no_grad():
        hubble_fcnn = quintessence.H_quint(z, lambda_fcnn, Om_m_0_fcnn, H_0_fcnn, solution_fcnn)
        hubble_bbb = quintessence.H_quint(z, lambda_bbb, Om_m_0_bbb, H_0_bbb, bbb_sampler)
        hubble_nlm = quintessence.H_quint(z, lambda_nlm, Om_m_0_nlm, H_0_nlm, nlm_sampler)
        hubble_hmc = quintessence.H_quint(z, lambda_hmc, Om_m_0_hmc, H_0_hmc, hmc_sampler)

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

def plot_quintessence(force=False):
    data = get_plot_data(force=force)
    func_names = ["$x(N')$", "$y(N')$"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        for j in range(2):
            axes[j][i].axvspan(quintessence.coords_train_min[0], quintessence.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
            axes[j][i].plot(data["x"], data["numerical"][j], 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Numerical Solution')
            axes[j][i].plot(data["x"], data["FCNN"][j], "red", alpha=.6, label="Det Solution")
            axes[j][i].plot(data["x"], data[method][j][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-1*data[method][j][1].ravel(), data[method][j][0].ravel()+1*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-2*data[method][j][1].ravel(), data[method][j][0].ravel()+2*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-3*data[method][j][1].ravel(), data[method][j][0].ravel()+3*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

            axes[j][i].set_xlim(data["x"].min(), data["x"].max())
            if j == 0: axes[j][i].set_title(method)
            if j == 1: axes[j][i].set_xlabel("$N'$", size=26)
            if i == 0: axes[j][i].set_ylabel(func_names[j], size=26)
            axes[j][i].set_ylim(-0.05, 0.5 if j == 0 else 0.9)

    fig.suptitle("Quintessence Forward Solutions", size=26, y=1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08))
    plt.savefig(f"figures/quintessence/forward_quintessence.png", bbox_inches='tight')

def plot_quintessence_bundle_errors(method, force=False):
    data = get_bundle_plot_data(force=force)

    error_fn = error_metrics[method]
    fcnn_errors = [error_fn(data["FCNN"][i], None, data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]
    nlm_errors = [error_fn(data["NLM"][i][0], data["NLM"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]
    bbb_errors = [error_fn(data["BBB"][i][0], data["BBB"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]
    hmc_errors = [error_fn(data["HMC"][i][0], data["HMC"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    lam_min, lam_max = data["x"][1].min(), data["x"][1].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(2, 4, figsize=(24, 8), sharex=True)
        im1 = ax[0, 0].imshow(fcnn_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im2 = ax[0, 1].imshow(nlm_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 2].imshow(bbb_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im4 = ax[0, 3].imshow(hmc_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        im5 = ax[1, 0].imshow(fcnn_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im6 = ax[1, 1].imshow(nlm_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 2].imshow(bbb_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im8 = ax[1, 3].imshow(hmc_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        ax[0, 0].text(-0.3, .5 , "$x$", usetex=True, va="center", size=26)
        ax[1, 0].text(-0.3, .5 , "$y$", usetex=True, va="center", size=26)

        fig.suptitle("Quintessence Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=1.05)
        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[1, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$\lambda$", size=20)
            ax[1, i].set_ylabel("$\lambda$", size=20)
            ax[0, i].set_title(m, size=20)

        fig.colorbar(im1, ax=ax[0, 0])
        fig.colorbar(im2, ax=ax[0, 1])
        fig.colorbar(im3, ax=ax[0, 2])
        fig.colorbar(im4, ax=ax[0, 3])

        fig.colorbar(im5, ax=ax[1, 0])
        fig.colorbar(im6, ax=ax[1, 1])
        fig.colorbar(im7, ax=ax[1, 2])
        fig.colorbar(im8, ax=ax[1, 3])
        fig.savefig(f"figures/quintessence/bundle_error_quintessence_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(2, 4, figsize=(24, 8), sharex=True)
        im1 = ax[0, 0].imshow(fcnn_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im2 = ax[0, 1].imshow(nlm_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 2].imshow(bbb_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 3].imshow(hmc_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        im5 = ax[1, 0].imshow(fcnn_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im6 = ax[1, 1].imshow(nlm_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 2].imshow(bbb_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 3].imshow(hmc_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        ax[0, 0].text(-0.15, .5 , "$x$", usetex=True, va="center", size=26)
        ax[1, 0].text(-0.15, .5 , "$y$", usetex=True, va="center", size=26)

        fig.suptitle("Quintessence Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=1.05)

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[1, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$\lambda$", size=20)
            ax[1, i].set_ylabel("$\lambda$", size=20)
            ax[0, i].set_title(m, size=20)

        fig.colorbar(im1, ax=ax[0, 0])
        fig.colorbar(im2, ax=ax[0, 1])
        fig.colorbar(im3, ax=ax[0, 2])
        fig.colorbar(im4, ax=ax[0, 3])

        fig.colorbar(im5, ax=ax[1, 0])
        fig.colorbar(im6, ax=ax[1, 1])
        fig.colorbar(im7, ax=ax[1, 2])
        fig.colorbar(im8, ax=ax[1, 3])
        fig.savefig(f"figures/quintessence/bundle_error_quintessence_{method}_train_{Om}.png", bbox_inches='tight')

def plot_quintessence_bundle_std_errors(method, force=False):
    data = get_bundle_plot_data(force=force)
    method = "std_" + method
    error_fn = error_metrics[method]
    nlm_errors = [error_fn(data["NLM"][i][0], data["NLM"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]
    bbb_errors = [error_fn(data["BBB"][i][0], data["BBB"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]
    hmc_errors = [error_fn(data["HMC"][i][0], data["HMC"][i][1], data["numerical"][i]).reshape(quintessence.bundle_plot_dimension_sizes) for i in range(2)]

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    lam_min, lam_max = data["x"][1].min(), data["x"][1].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        im2 = ax[0, 0].imshow(nlm_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 1].imshow(bbb_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im4 = ax[0, 2].imshow(hmc_errors[0][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        im6 = ax[1, 0].imshow(nlm_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 1].imshow(bbb_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im8 = ax[1, 2].imshow(hmc_errors[1][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        ax[0, 0].text(-0.3, .5 , "$x$", usetex=True, va="center", size=26)
        ax[1, 0].text(-0.3, .5 , "$y$", usetex=True, va="center", size=26)

        fig.suptitle("Quintessence Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om]}$ (Test Region)", size=26, y=1.05)
        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[1, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$\lambda$", size=20)
            ax[1, i].set_ylabel("$\lambda$", size=20)
            ax[0, i].set_title(m, size=20)

        fig.colorbar(im2, ax=ax[0, 0])
        fig.colorbar(im3, ax=ax[0, 1])
        fig.colorbar(im4, ax=ax[0, 2])

        fig.colorbar(im6, ax=ax[1, 0])
        fig.colorbar(im7, ax=ax[1, 1])
        fig.colorbar(im8, ax=ax[1, 2])
        fig.savefig(f"figures/quintessence/bundle_std_error_quintessence_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
        im2 = ax[0, 0].imshow(nlm_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 1].imshow(bbb_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[0, 2].imshow(hmc_errors[0][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        im6 = ax[1, 0].imshow(nlm_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 1].imshow(bbb_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im7 = ax[1, 2].imshow(hmc_errors[1][:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        ax[0, 0].text(-0.15, .5 , "$x$", usetex=True, va="center", size=26)
        ax[1, 0].text(-0.15, .5 , "$y$", usetex=True, va="center", size=26)

        fig.suptitle("Quintessence Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om]}$ (Train Region)", size=26, y=1.05)

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[1, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$\lambda$", size=20)
            ax[1, i].set_ylabel("$\lambda$", size=20)
            ax[0, i].set_title(m, size=20)

        fig.colorbar(im2, ax=ax[0, 0])
        fig.colorbar(im3, ax=ax[0, 1])
        fig.colorbar(im4, ax=ax[0, 2])

        fig.colorbar(im6, ax=ax[1, 0])
        fig.colorbar(im7, ax=ax[1, 1])
        fig.colorbar(im8, ax=ax[1, 2])
        fig.savefig(f"figures/quintessence/bundle_std_error_quintessence_{method}_train_{Om}.png", bbox_inches='tight')

def plot_bundle_examples(force=False):
    data = get_bundle_plot_data(force=force)
    x = data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(2, 3, figsize=(18, 8))
        for i in map(int, np.linspace(0, quintessence.bundle_plot_dimension_sizes[1]-1, 5)):
            im1 = ax[0, 0].plot(x[:, 0, Om], data["FCNN"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im2 = ax[0, 1].plot(x[:, 0, Om], data["NLM"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im3 = ax[0, 2].plot(x[:, 0, Om], data["BBB"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im4 = ax[0, 3].plot(x[:, 0, Om], data["HMC"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            
            ax[0, 0].plot(x[:, 0], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 1].plot(x[:, 0], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 2].plot(x[:, 0], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 3].plot(x[:, 0], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())

            im1 = ax[1, 0].plot(x[:, 0, Om], data["FCNN"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im2 = ax[1, 1].plot(x[:, 0, Om], data["NLM"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im3 = ax[1, 2].plot(x[:, 0, Om], data["BBB"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im4 = ax[1, 3].plot(x[:, 0, Om], data["HMC"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            
            ax[1, 0].plot(x[:, 0], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 1].plot(x[:, 0], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 2].plot(x[:, 0], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 3].plot(x[:, 0], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[0, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$x$", size=20)
            ax[0, i].set_title(m, size=20)
            ax[0, i].axvspan(x[:, 0, Om][0], x[:, 0, Om][int(quintessence.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')
            ax[0, i].set_xlim(z_min, z_max)
            ax[1, i].set_xlabel("$z$", size=20)
            ax[1, i].set_ylabel("$y$", size=20)
            ax[1, i].set_xlim(z_min, z_max)
            ax[1, i].axvspan(x[:, 0, Om][0], x[:, 0, Om][int(quintessence.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')

        handles, labels = ax[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle("Quintessence solutions for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$", size=26, y=1.05)
        plt.savefig(f"figures/quintessence/bundle_examples_quintessence_{Om}.png", bbox_inches='tight')

def plot_hubble_forward(force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_plot_data(force=force)

    hubble_an = quintessence.H_quint(data["x"], 1, .3, 65, data["numerical"])
    hubble_fcnn = quintessence.H_quint(data["x"], 1, .3, 65, data["FCNN"])
    hubble_bbb = quintessence.H_quint(data["x"], 1, .3, 65, data["BBB_samples"])
    hubble_nlm = quintessence.H_quint(data["x"], 1, .3, 65, data["NLM_samples"])
    hubble_hmc = quintessence.H_quint(data["x"], 1, .3, 65, data["HMC_samples"])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(quintessence.coords_train_min[0], quintessence.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
    plt.suptitle("Quintessence Hubble Forward", y=1.05, size=26)
    plt.savefig(f"figures/quintessence/hubble_quintessence{nan_text}.png", bbox_inches='tight')

def plot_hubble_best_fit(force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_plot_data(force=force)

    lam_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_quintessence_fcnn_cc.npy").mean(axis=0)
    lam_bbb, Om_m_0_bbb, H_0_bbb = np.load("checkpoints/inverse_samples_bundle_quintessence_bbb_cc.npy").mean(axis=0)
    lam_nlm, Om_m_0_nlm, H_0_nlm = np.load("checkpoints/inverse_samples_bundle_quintessence_nlm_cc.npy").mean(axis=0)
    lam_hmc, Om_m_0_hmc, H_0_hmc = np.load("checkpoints/inverse_samples_bundle_quintessence_hmc_cc.npy").mean(axis=0)

    hubble_fcnn = quintessence.H_quint(data["x"],lam_fcnn, Om_m_0_fcnn, H_0_fcnn, data["FCNN"])
    hubble_bbb = quintessence.H_quint(data["x"],lam_bbb, Om_m_0_bbb, H_0_bbb, data["BBB_samples"])
    hubble_nlm = quintessence.H_quint(data["x"],lam_nlm, Om_m_0_nlm, H_0_nlm, data["NLM_samples"])
    hubble_hmc = quintessence.H_quint(data["x"],lam_hmc, Om_m_0_hmc, H_0_hmc, data["HMC_samples"])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(quintessence.coords_train_min[0], quintessence.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
    plt.suptitle("Quintessence Hubble Best Fit", y=1.05, size=26)
    plt.savefig(f"figures/quintessence/hubble_quintessence_best_fit{nan_text}.png", bbox_inches='tight')

def plot_bundle_examples():
    data = get_bundle_plot_data()
    z_min, z_max = data["x"][0].min(), data["x"][0].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)
        fig, ax = plt.subplots(2, 4, figsize=(24, 8))
        for i in map(int, np.linspace(0, quintessence.bundle_plot_dimension_sizes[1]-1, 5)):
            im1 = ax[0, 0].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["FCNN"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im2 = ax[0, 1].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im3 = ax[0, 2].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im4 = ax[0, 3].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")

            ax[0, 1].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["NLM"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["NLM"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["NLM"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[0, 2].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["BBB"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["BBB"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["BBB"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[0, 3].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["HMC"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["HMC"][0][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["HMC"][0][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            
            ax[0, 0].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 1].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 2].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[0, 3].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())

            im1 = ax[1, 0].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["FCNN"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im2 = ax[1, 1].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im3 = ax[1, 2].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")
            im4 = ax[1, 3].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(quintessence.bundle_plot_dimension_sizes)[0, i, Om], 2)}$")

            ax[1, 1].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["NLM"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["NLM"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["NLM"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[1, 2].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["BBB"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["BBB"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["BBB"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            ax[1, 3].fill_between(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]-2*data["HMC"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], data["HMC"][1][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om]+2*data["HMC"][1][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
            
            ax[1, 0].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 1].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 2].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())
            ax[1, 3].plot(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][1].reshape(quintessence.bundle_plot_dimension_sizes)[:, i, Om], "--", color=im1[0].get_color())

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[0, i].set_xlabel("$z$", size=20)
            ax[0, i].set_ylabel("$x$", size=20)
            ax[0, i].set_title(m, size=20)
            ax[0, i].axvspan(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om][0], data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om][int(quintessence.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')
            ax[0, i].set_xlim(z_min, z_max)
            ax[1, i].set_xlabel("$z$", size=20)
            ax[1, i].set_ylabel("$y$", size=20)
            ax[1, i].set_xlim(z_min, z_max)
            ax[1, i].axvspan(data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om][0], data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes)[:, 0, Om][int(quintessence.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')

        handles, labels = ax[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle("Quintessence Bundle Solutions for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$", size=26, y=1.)
        plt.savefig(f"figures/quintessence/bundle_examples_quintessence_{Om}.png", bbox_inches='tight')

def plot_hubble_bundle_errors(method, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_bundle_plot_data(force=force)
    z, lam, Om = data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes), data["x"][1].reshape(quintessence.bundle_plot_dimension_sizes), data["x"][2].reshape(quintessence.bundle_plot_dimension_sizes)

    hubble_an = quintessence.H_quint(z, lam, Om, 65, data["numerical"])
    hubble_fcnn = quintessence.H_quint(z, lam, Om, 65, data["FCNN"])
    hubble_bbb = quintessence.H_quint(z, lam, Om, 65, [data["BBB_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_nlm = quintessence.H_quint(z, lam, Om, 65, [data["NLM_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_hmc = quintessence.H_quint(z, lam, Om, 65, [data["HMC_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])

    hubble_bbb_mean = np.__dict__[mean_fn](hubble_bbb, axis=0)
    hubble_nlm_mean = np.__dict__[mean_fn](hubble_nlm, axis=0)
    hubble_hmc_mean = np.__dict__[mean_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(hubble_fcnn, None, hubble_an)
    bbb_errors = error_fn(hubble_bbb_mean, None, hubble_an)
    nlm_errors = error_fn(hubble_nlm_mean, None, hubble_an)
    hmc_errors = error_fn(hubble_hmc_mean, None, hubble_an)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    lam_min, lam_max = data["x"][1].min(), data["x"][1].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(1, 4, figsize=(24, 4), sharex=True)
        im1 = ax[0].imshow(fcnn_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        fig.suptitle("Quintessence Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=1.05)
        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$H_{quint}$", size=20)
            ax[i].set_title(m, size=20)

        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        fig.colorbar(im4, ax=ax[3])
        fig.savefig(f"figures/quintessence/hubble_bundle_error_quintessence{nan_text}_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 4, figsize=(24, 4), sharex=True)
        im1 = ax[0].imshow(fcnn_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        fig.suptitle("Quintessence Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=1.05)

        for i, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$H_{quint}$", size=20)
            ax[i].set_title(m, size=20)

        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        fig.colorbar(im4, ax=ax[3])
        fig.savefig(f"figures/quintessence/hubble_bundle_error_quintessence{nan_text}_{method}_train_{Om}.png", bbox_inches='tight')

def plot_hubble_bundle_std_errors(method, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    method = "std_" + method

    data = get_bundle_plot_data(force=force)
    z, lam, Om = data["x"][0].reshape(quintessence.bundle_plot_dimension_sizes), data["x"][1].reshape(quintessence.bundle_plot_dimension_sizes), data["x"][2].reshape(quintessence.bundle_plot_dimension_sizes)
    hubble_an = quintessence.H_quint(z, lam, Om, 65, data["numerical"])
    hubble_bbb = quintessence.H_quint(z, lam, Om, 65, [data["BBB_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_nlm = quintessence.H_quint(z, lam, Om, 65, [data["NLM_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_hmc = quintessence.H_quint(z, lam, Om, 65, [data["HMC_samples"][i].reshape(-1, *quintessence.bundle_plot_dimension_sizes) for i in range(2)])

    hubble_bbb_mean, hubble_bbb_std = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    hubble_nlm_mean, hubble_nlm_std = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    hubble_hmc_mean, hubble_hmc_std = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    bbb_errors = error_fn(hubble_bbb_mean, hubble_bbb_std, hubble_an).reshape(quintessence.bundle_plot_dimension_sizes)
    nlm_errors = error_fn(hubble_nlm_mean, hubble_nlm_std, hubble_an).reshape(quintessence.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(hubble_hmc_mean, hubble_hmc_std, hubble_an).reshape(quintessence.bundle_plot_dimension_sizes)

    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    lam_min, lam_max = data["x"][1].min(), data["x"][1].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
        im2 = ax[0].imshow(nlm_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, lam_min, lam_max], aspect="auto")

        fig.suptitle("Quintessence Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=1.05)
        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$H_{quint}$", size=20)
            ax[i].set_title(m, size=20)

        fig.colorbar(im2, ax=ax[0])
        fig.colorbar(im3, ax=ax[1])
        fig.colorbar(im4, ax=ax[2])
        fig.savefig(f"figures/quintessence/hubble_bundle_std_error_quintessence{nan_text}_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
        im2 = ax[0].imshow(nlm_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, :, Om].T[:int(quintessence.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, lam_min, lam_max], aspect="auto")

        fig.suptitle("Quintessence Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(quintessence.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=1.05)

        for i, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[i].set_xlabel("$z$", size=20)
            ax[i].set_ylabel("$H_{quint}$", size=20)
            ax[i].set_title(m, size=20)

        fig.colorbar(im2, ax=ax[0])
        fig.colorbar(im3, ax=ax[1])
        fig.colorbar(im4, ax=ax[2])
        fig.savefig(f"figures/quintessence/hubble_bundle_std_error_quintessence{nan_text}_{method}_train_{Om}.png", bbox_inches='tight')

def plot_calibration(data, name):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i in range(2):
        uct.plot_calibration(data["BBB"][i][0].ravel(), data["BBB"][i][1].ravel(), data["numerical"][i].ravel(), ax=axes[i, 0])
        uct.plot_calibration(data["NLM"][i][0].numpy().ravel(), data["NLM"][i][1].numpy().ravel(), data["numerical"][i].ravel(), ax=axes[i, 1])
        uct.plot_calibration(data["HMC"][i][0].ravel(), data["HMC"][i][1].ravel(), data["numerical"][i].ravel(), ax=axes[i, 2])
        axes[i, 0].set_title("BBB", size=26)
        axes[i, 1].set_title("NLM", size=26)
        axes[i, 2].set_title("HMC", size=26)

    fig.suptitle("Quintessence Forward Solution Calibration", size=26, y=1.05)
    plt.savefig(f"figures/quintessence/{name}.png", bbox_inches='tight')

def plot_calibration_dts(data_train, data_test, data_ood, name):
    method = "Bundle" if "bundle" in name else "Forward"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    labels = ["Training Domain", "Testing Domain", "OOD Domain"]
    for i in range(2):
        print("BBB")
        plot_calibration_area(
            [data["BBB"][i][0].ravel() for data in [data_train, data_test, data_ood]],
            [data["BBB"][i][1].ravel() for data in [data_train, data_test, data_ood]],
            [data["numerical"][i].ravel() for data in [data_train, data_test, data_ood]],
            ax=axes[i, 0],
            labels=labels
        )
        print("NLM")
        plot_calibration_area(
            [data["NLM"][i][0].numpy().ravel() for data in [data_train, data_test, data_ood]],
            [data["NLM"][i][1].numpy().ravel() for data in [data_train, data_test, data_ood]],
            [data["numerical"][i].ravel() for data in [data_train, data_test, data_ood]],
            ax=axes[i, 1],
            labels=labels
        )
        print("HMC")
        plot_calibration_area(
            [data["HMC"][i][0].ravel() for data in [data_train, data_test, data_ood]],
            [data["HMC"][i][1].ravel() for data in [data_train, data_test, data_ood]],
            [data["numerical"][i].ravel() for data in [data_train, data_test, data_ood]],
            ax=axes[i, 2],
            labels=labels
        )
    # axes[0, 0].set_title("BBB", size=26)
    # axes[0, 1].set_title("NLM", size=26)
    # axes[0, 2].set_title("HMC", size=26)
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[0, 2].set_xlabel("")
    axes[1, 0].set_title("")
    axes[1, 1].set_title("")
    axes[1, 2].set_title("")

    axes[0, 0].annotate("BBB", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[0, 1].annotate("NLM", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[0, 2].annotate("HMC", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')

    axes[0, 0].annotate("$x$", xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - 5, 0), xycoords=axes[0, 0].yaxis.label, textcoords='offset points', size=26, ha='right', va='center')
    axes[1, 0].annotate("$y$", xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - 5, 0), xycoords=axes[1, 0].yaxis.label, textcoords='offset points', size=26, ha='right', va='center')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Quintessence {method} Solution Calibration", size=26, y=1)
    plt.savefig(f"figures/quintessence/{name}.png", bbox_inches='tight')

if __name__ == "__main__":
    # plot_quintessence()
    # plot_hubble_forward()
    # plot_hubble_forward(ignore_nans=True)
    # plot_hubble_best_fit()
    # plot_hubble_best_fit(ignore_nans=True)
    # plot_quintessence_bundle_errors(method="ae")
    # plot_quintessence_bundle_errors(method="re")
    # plot_quintessence_bundle_errors(method="rpd")
    # plot_quintessence_bundle_std_errors(method="ae")
    # plot_quintessence_bundle_std_errors(method="re")
    # plot_quintessence_bundle_std_errors(method="rpd")
    # plot_bundle_examples()
    # plot_hubble_bundle_errors(method="ae")
    # plot_hubble_bundle_errors(method="re")
    # plot_hubble_bundle_errors(method="rpd")
    # plot_hubble_bundle_errors(method="ae", ignore_nans=True)
    # plot_hubble_bundle_errors(method="re", ignore_nans=True)
    # plot_hubble_bundle_errors(method="rpd", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="ae")
    # plot_hubble_bundle_std_errors(method="re")
    # plot_hubble_bundle_std_errors(method="rpd")
    # plot_hubble_bundle_std_errors(method="ae", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="re", ignore_nans=True)
    # plot_hubble_bundle_std_errors(method="rpd", ignore_nans=True)

    # for dt in ["test", "train", "ood"]:
    #     plot_calibration(get_plot_data(domain_type=dt), f"calibration_quintessence_forward_{dt}")
    #     plot_calibration(get_bundle_plot_data(domain_type=dt), f"calibration_quintessence_bundle_{dt}")

    plot_calibration_dts(*[get_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_quintessence_forward_all")
    plot_calibration_dts(*[get_bundle_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_quintessence_bundle_all")