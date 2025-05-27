from gc import collect
import numpy as np
import dill
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from neurodiffeq.solvers import Solver1D, BundleSolver1D
import torch
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
import equations.hs as hs
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

def numerical_hs(t, b_prime=hs.b_prime, Om_m_0=hs.Om_m_0):
    def dX_dz_f_R(z_prime, variables, b):
        z = hs.z_0 * (1 - z_prime)

        x = variables[0]
        y = variables[1]
        v = variables[2]
        Om = variables[3]
        r = variables[4]

        N = (r + b)*(((r + b)**2) - 2*b)
        D = 4*b*r
        Gamma = N/D

        s0 = -hs.z_0*(-Om + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = -hs.z_0*(- (v*x*Gamma - x*y + 4*y - 2*y*v)) / (z+1)
        s2 = -hs.z_0*(-v * (x*Gamma + 4 - 2*v)) / (z+1)
        s3 = -hs.z_0*(Om * (-1 + x + 2*v)) / (z+1)
        s4 = -hs.z_0*(-(x * r * Gamma)) / (1+z)

        return [s0, s1, s2, s3, s4]


    def x_0():
        x_0 = 0.0
        return x_0


    def y_0():
        z = hs.z_0 * (1 - hs.z_prime_0)
        y_0 = (Om_m_0*((1 + z)**3) + 2*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + (1 - Om_m_0)))
        return y_0


    def v_0():
        z = hs.z_0 * (1 - hs.z_prime_0)
        return (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(2*(Om_m_0*((1 + z)**3) + (1 - Om_m_0)))


    def Om_0():
        z = hs.z_0 * (1 - hs.z_prime_0)
        Om_0 = Om_m_0*((1 + z)**3)/((Om_m_0*((1 + z)**3) + (1 - Om_m_0)))
        return Om_0


    def r_0():
        z = hs.z_0 * (1 - hs.z_prime_0)
        r_0 = (Om_m_0*((1 + z)**3) + 4*(1 - Om_m_0))/(1 - Om_m_0)
        return r_0


    def sol_r(z_prime, b, Om_m_0):
        cond_Num_r = [x_0(),
                    y_0(),
                    v_0(),
                    Om_0(),
                    r_0()]

        sol2_r = solve_ivp(dX_dz_f_R, (0, 1), cond_Num_r,
                        rtol=1e-11, atol=1e-16,
                        #rtol=1e-6, atol=1e-9,
                        t_eval=t.ravel(),
                        args=(b,)
                        )
        solutions = sol2_r.y
        solutions[-1] = np.log(solutions[-1]) # transform r to r' = log(r)
        return solutions
    return sol_r(t, hs.b_max*b_prime, Om_m_0)

def numerical_hs_bundle(t, b_prime, Om_m_0):
    solutions = [np.zeros((t.shape[0], b_prime.shape[0], Om_m_0.shape[0])) for _ in range(5)]
    for i, b in tqdm(enumerate(b_prime.ravel()), desc="Bundle Numerical b Prime", total=b_prime.shape[0]):
        for j, o in tqdm(enumerate(Om_m_0.ravel()), leave=False, desc="Bundle Numerical Om_m_0", total=Om_m_0.shape[0]):
            for k, sol in enumerate(numerical_hs(t, b, o)):
                solutions[k][:, i, j] = sol.ravel()
    return solutions

def get_bundle_plot_data_fcnn(x_test):
    print("Loading FCNN")
    solver = BundleSolver1D.load("checkpoints/solver_bundle_hs_clfcnn.ndeq")
    solution = solver.get_solution()(*x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(*x_test, to_numpy=True)
    return solution, fcnn_residuals

def get_bundle_plot_data_bbb(x_test, batches, batch_size):
    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_bundle_hs_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = hs.system_bundle
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x) for x in x_test], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp)
        bbb_residuals.append(bbb_residuals_tmp)
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(5)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(5)]
    return bbb_samples, bbb_residuals

def get_bundle_plot_data_nlm(x_test):
    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_bundle_hs_nlm.pt")
    nlm_model.diff_eqs = hs.system_bundle
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_means, nlm_stds = nlm_model.posterior_predictive(x_test, include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive(x_test, n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals(x_test)
    return nlm_means, nlm_stds, nlm_samples, nlm_residuals

def get_bundle_plot_data_hmc(x_test, batch_size):
    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_bundle_hs_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = hs.system_bundle
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_hs_hmc.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x) for x in x_test], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp)
        hmc_residuals.append(hmc_residuals_tmp)
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(5)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(5)]
    return hmc_samples, hmc_residuals

@dill_dec("hs")
def get_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        x_test = np.linspace(hs.coords_test_min, hs.coords_test_max, 200).reshape(-1, 1)
    elif domain_type == "train":
        x_test = np.linspace(hs.coords_train_min, hs.coords_train_max, 200).reshape(-1, 1)
    else:
        x_test = np.linspace(hs.coords_test_min, hs.coords_train_min, 200).reshape(-1, 1)

    numerical = [a.reshape(-1, 1) for a in numerical_hs(x_test)]
    batch_size = 10
    batches = 10_000 // batch_size

    print("Loading FCNN")
    solver = Solver1D.load("checkpoints/solver_hs_clfcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(x_test, to_numpy=True)

    print("Loading BBB")
    solver_bbb = torch.load(f"checkpoints/solver_hs_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = hs.system
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x_test)], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp)
        bbb_residuals.append(bbb_residuals_tmp)
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(5)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(5)]

    print("Loading NLM")
    nlm_model = NLMModel.load(f"checkpoints/model_hs_nlm.pt")
    nlm_model.diff_eqs = hs.system
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_means, nlm_stds = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([x_test], n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals([x_test])

    print("Loading HMC")
    hmc_solver = torch.load(f"checkpoints/solver_hs_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = hs.system
    hmc_posterior_samples = torch.load(f"checkpoints/samples_hs_hmc.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x_test)], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp)
        hmc_residuals.append(hmc_residuals_tmp)
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(5)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(5)]

    data = {
        "domain_type": domain_type,
         "x": x_test,
         "numerical": numerical,
         "FCNN": solution,
         "BBB": [(bbb_samples[i].mean(axis=0), bbb_samples[i].std(axis=0)) for i in range(5)],
         "NLM": [(nlm_means[i].detach().cpu(), nlm_stds[i].detach().cpu()) for i in range(5)],
         "HMC": [(hmc_samples[i].mean(axis=0), hmc_samples[i].std(axis=0)) for i in range(5)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": hmc_residuals,
         }
    return data

@dill_dec("hs", bundle=True)
def get_bundle_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        t_test = np.linspace(hs.coords_test_min, hs.coords_test_max, hs.bundle_plot_dimension_sizes[0])
    elif domain_type == "train":
        t_test = np.linspace(hs.coords_train_min, hs.coords_train_max, hs.bundle_plot_dimension_sizes[0])
    else:
        t_test = np.linspace(hs.coords_test_min, hs.coords_train_min, hs.bundle_plot_dimension_sizes[0])

    params_test = [np.linspace(hs.bundle_parameters_min_plot[i], hs.bundle_parameters_max_plot[i], hs.bundle_plot_dimension_sizes[i+1]) for i in range(len(hs.bundle_parameters_min))]
    x_test = [x.reshape(-1, 1) for x in np.meshgrid(t_test, *params_test, indexing="ij")]
    numerical = numerical_hs_bundle(t_test, *params_test)

    batch_size = 10
    batches = 10_000 // batch_size

    solution, fcnn_residuals = get_bundle_plot_data_fcnn(x_test)
    collect()
    bbb_samples, bbb_residuals= get_bundle_plot_data_bbb(x_test, batches, batch_size)
    collect()
    nlm_means, nlm_stds, nlm_samples, nlm_residuals = get_bundle_plot_data_nlm(x_test)
    collect()
    hmc_samples, hmc_residuals = get_bundle_plot_data_hmc(x_test, batch_size)
    collect()

    data = {
        "domain_type": domain_type,
         "x": x_test,
         "numerical": [num.reshape(hs.bundle_plot_dimension_sizes) for num in numerical],
         "FCNN": [sol.reshape(hs.bundle_plot_dimension_sizes) for sol in solution],
         "BBB": [(bbb_samples[i].mean(axis=0).reshape(hs.bundle_plot_dimension_sizes), bbb_samples[i].std(axis=0).reshape(hs.bundle_plot_dimension_sizes)) for i in range(5)],
         "NLM": [(nlm_means[i].detach().cpu().reshape(hs.bundle_plot_dimension_sizes), nlm_stds[i].detach().cpu().reshape(hs.bundle_plot_dimension_sizes)) for i in range(5)],
         "HMC": [(hmc_samples[i].mean(axis=0).reshape(hs.bundle_plot_dimension_sizes), hmc_samples[i].std(axis=0).reshape(hs.bundle_plot_dimension_sizes)) for i in range(5)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": hmc_residuals,
         }
    return data

@dill_dec_old("plot_data/hs_best_fit.dill", "plot_data/hs_best_fit_eb.dill")
def get_best_fit_plot_data():
    b_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_hs_clfcnn_cc.npy").mean(axis=0)
    b_bbb, Om_m_0_bbb, H_0_bbb = np.load(f"checkpoints/inverse_samples_bundle_hs_bbb_cc.npy").mean(axis=0)
    b_nlm, Om_m_0_nlm, H_0_nlm = np.load(f"checkpoints/inverse_samples_bundle_hs_nlm_cc.npy").mean(axis=0)
    b_hmc, Om_m_0_hmc, H_0_hmc = np.load(f"checkpoints/inverse_samples_bundle_hs_hmc_cc.npy").mean(axis=0)

    z = np.linspace(0, 2, 100).reshape(-1, 1)

    solver_fcnn = BundleSolver1D.load("checkpoints/solver_bundle_hs_clfcnn.ndeq")
    solution_fcnn = solver_fcnn.get_solution()

    bbb_solver = torch.load(f"checkpoints/solver_bundle_hs_bbb.pyro", map_location="cpu")
    if "get_likelihood_std" in bbb_solver.__dict__:
        bbb_solver.get_likelihood_std.device = "cpu"

    nlm_model = NLMModel.load(f"checkpoints/model_bundle_hs_nlm.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"

    hmc_solver = torch.load(f"checkpoints/solver_bundle_hs_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.device = "cpu"
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_hs_hmc.pyro", pickle_module=dill)

    bbb_sampler = bbb_sample_solutions(bbb_solver, 10_000, 2)
    nlm_sampler = nlm_sample_solutions(nlm_model, 10_000, 2)
    hmc_sampler = hmc_sample_solutions(hmc_solver, hmc_posterior_samples, 10_000, 2)

    with torch.no_grad():
        hubble_fcnn = hs.H_HS(z, b_fcnn, Om_m_0_fcnn, H_0_fcnn, solution_fcnn)
        hubble_bbb = hs.H_HS(z, b_bbb, Om_m_0_bbb, H_0_bbb, bbb_sampler)
        hubble_nlm = hs.H_HS(z, b_nlm, Om_m_0_nlm, H_0_nlm, nlm_sampler)
        hubble_hmc = hs.H_HS(z, b_hmc, Om_m_0_hmc, H_0_hmc, hmc_sampler)

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

def plot_hs(force=False):
    data = get_plot_data(force=force)
    func_names = ["$x$", "$y$", "$v$", "$\Omega$", "$r$"]
    y_lims = [(-0.3216285014124655, 0.01531564292440312), 
          (0.47285657650932067, 0.9208326653852308),
          (0.4598577410423105, 1.2485332207483506),
          (0.3506590119785351, 1.028565034554888),
          (0.7057214599234863, 6.361112119661397)]

    fig, axes = plt.subplots(5, 3, figsize=(18, 20), sharex=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        for j in range(5):
            axes[j][i].axvspan(hs.coords_train_min[0], hs.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
            axes[j][i].plot(data["x"], data["numerical"][j], 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Numerical Solution')
            axes[j][i].plot(data["x"], data["FCNN"][j], "red", alpha=.6, label="Det Solution")
            axes[j][i].plot(data["x"], data[method][j][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-1*data[method][j][1].ravel(), data[method][j][0].ravel()+1*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-2*data[method][j][1].ravel(), data[method][j][0].ravel()+2*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
            axes[j][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-3*data[method][j][1].ravel(), data[method][j][0].ravel()+3*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

            axes[j][i].set_xlim(data["x"].min(), data["x"].max())
            if j == 0: axes[j][i].set_title(method)
            if j == 4: axes[j][i].set_xlabel("$z$", size=26)
            if i == 0: axes[j][i].set_ylabel(func_names[j], size=26)
            axes[j][i].set_ylim(*y_lims[j])

    fig.suptitle("HS Forward Solutions", size=26, y=.92)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.04))
    plt.savefig(f"figures/hs/forward_hs.png", bbox_inches='tight')

def plot_hs_bundle_errors(method, force=False):
    data = get_bundle_plot_data(force=force)
    func_names = ["$x$", "$y$", "$v$", "$\Omega$", "$r$"]
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    b_min, b_max = data["x"][2].min(), data["x"][2].max()

    error_fn = error_metrics[method]
    fcnn_errors = [error_fn(data["FCNN"][i], None, data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]
    nlm_errors = [error_fn(data["NLM"][i][0], data["NLM"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]
    bbb_errors = [error_fn(data["BBB"][i][0], data["BBB"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]
    hmc_errors = [error_fn(data["HMC"][i][0], data["HMC"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(5, 4, figsize=(24, 20))
        for i in range(5):
            im1 = ax[i, 0].imshow(fcnn_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
            im2 = ax[i, 1].imshow(nlm_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
            im3 = ax[i, 2].imshow(bbb_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
            im4 = ax[i, 3].imshow(hmc_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")

            fig.colorbar(im1, ax=ax[i, 0])
            fig.colorbar(im2, ax=ax[i, 1])
            fig.colorbar(im3, ax=ax[i, 2])
            fig.colorbar(im4, ax=ax[i, 3])
            ax[i, 0].text(-0.3, .25 , func_names[i], usetex=True, va="center", size=26)

            for j, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
                ax[i, j].set_ylabel("$b$", size=20)
                ax[0, j].set_title(m, size=20)
                
        for i in range(4): ax[-1, i].set_xlabel("$z$", size=20)
        fig.suptitle("HS Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=.92)
        fig.savefig(f"figures/hs/bundle_error_hs_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(5, 4, figsize=(24, 20))
        for i in range(5):
            im1 = ax[i, 0].imshow(fcnn_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
            im2 = ax[i, 1].imshow(nlm_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
            im3 = ax[i, 2].imshow(bbb_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
            im3 = ax[i, 3].imshow(hmc_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")

            fig.colorbar(im1, ax=ax[i, 0])
            fig.colorbar(im2, ax=ax[i, 1])
            fig.colorbar(im3, ax=ax[i, 2])
            fig.colorbar(im4, ax=ax[i, 3])
            ax[i, 0].text(-0.15, .25 , func_names[i], usetex=True, va="center", size=26)

            for j, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
                ax[i, j].set_ylabel("$b$", size=20)
                ax[0, j].set_title(m, size=20)
                
        for i in range(4): ax[-1, i].set_xlabel("$z$", size=20)
        fig.suptitle("HS Solution " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=.92)

        fig.savefig(f"figures/hs/bundle_error_hs_{method}_train_{Om}.png", bbox_inches='tight')

def plot_hs_bundle_std_errors(method="re", force=False):
    data = get_bundle_plot_data(force=force)
    func_names = ["$x$", "$y$", "$v$", "$\Omega$", "$r$"]
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    b_min, b_max = data["x"][2].min(), data["x"][2].max()

    method = "std_" + method
    error_fn = error_metrics[method]
    nlm_errors = [error_fn(data["NLM"][i][0], data["NLM"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]
    bbb_errors = [error_fn(data["BBB"][i][0], data["BBB"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]
    hmc_errors = [error_fn(data["HMC"][i][0], data["HMC"][i][1], data["numerical"][i]).reshape(hs.bundle_plot_dimension_sizes) for i in range(5)]

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(5, 3, figsize=(18, 20))
        for i in range(5):
            im2 = ax[i, 0].imshow(nlm_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
            im3 = ax[i, 1].imshow(bbb_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
            im4 = ax[i, 2].imshow(hmc_errors[i][:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")

            fig.colorbar(im2, ax=ax[i, 0])
            fig.colorbar(im3, ax=ax[i, 1])
            fig.colorbar(im4, ax=ax[i, 2])
            ax[i, 0].text(-0.3, .25 , func_names[i], usetex=True, va="center", size=26)

            for j, m in enumerate(["NLM", "BBB", "HMC"]):
                ax[i, j].set_ylabel("$b$", size=20)
                ax[0, j].set_title(m, size=20)
                
        for i in range(3): ax[-1, i].set_xlabel("$z$", size=20)
        fig.suptitle("HS Solution" + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=.92)
        fig.savefig(f"figures/hs/bundle_std_error_hs_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(5, 3, figsize=(18, 20))
        for i in range(5):
            im2 = ax[i, 0].imshow(nlm_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
            im3 = ax[i, 1].imshow(bbb_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
            im3 = ax[i, 2].imshow(hmc_errors[i][:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")

            fig.colorbar(im2, ax=ax[i, 0])
            fig.colorbar(im3, ax=ax[i, 1])
            fig.colorbar(im4, ax=ax[i, 2])
            ax[i, 0].text(-0.15, .25, func_names[i], usetex=True, va="center", size=26)

            for j, m in enumerate(["NLM", "BBB", "HMC"]):
                ax[i, j].set_ylabel("$b$", size=20)
                ax[0, j].set_title(m, size=20)
                
        for i in range(3): ax[-1, i].set_xlabel("$z$", size=20)
        fig.suptitle("HS Solution" + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=.92)

        fig.savefig(f"figures/hs/bundle_std_error_hs_{method}_train_{Om}.png", bbox_inches='tight')

def plot_bundle_examples(force=False):
    data = get_bundle_plot_data(force=force)
    func_names = ["$x$", "$y$", "$v$", "$\Omega$", "$r$"]
    z_min, z_max = data["x"][0].min(), data["x"][0].max()

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)
        fig, ax = plt.subplots(5, 4, figsize=(24, 20))

        for lam_idx in map(int, np.linspace(0, hs.bundle_plot_dimension_sizes[1]-1, 5)):
            for i in range(5):
                im1 = ax[i, 0].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["FCNN"][i].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(hs.bundle_plot_dimension_sizes)[0, lam_idx, Om], 2)}$")
                im2 = ax[i, 1].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(hs.bundle_plot_dimension_sizes)[0, lam_idx, Om], 2)}$")
                im3 = ax[i, 2].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(hs.bundle_plot_dimension_sizes)[0, lam_idx, Om], 2)}$")
                im4 = ax[i, 3].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], label="$\Omega_{m,0}=" + f"{round(data['x'][1].reshape(hs.bundle_plot_dimension_sizes)[0, lam_idx, Om], 2)}$")

                ax[i, 1].fill_between(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["NLM"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]-2*data["NLM"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], data["NLM"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]+2*data["NLM"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
                ax[i, 2].fill_between(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["BBB"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]-2*data["BBB"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], data["BBB"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]+2*data["BBB"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
                ax[i, 3].fill_between(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["HMC"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]-2*data["HMC"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], data["HMC"][i][0].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om]+2*data["HMC"][i][1].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], color=im1[0].get_color(), alpha=0.2, edgecolor=None)
                
                ax[i, 0].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][i].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], "--", color=im1[0].get_color())
                ax[i, 1].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][i].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], "--", color=im1[0].get_color())
                ax[i, 2].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][i].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], "--", color=im1[0].get_color())
                ax[i, 3].plot(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om], data["numerical"][i].reshape(hs.bundle_plot_dimension_sizes)[:, lam_idx, Om], "--", color=im1[0].get_color())

        for i in range(5):
            for j, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
                ax[i, j].set_ylabel(func_names[i], size=20)
                ax[0, j].set_title(m, size=20)
                ax[i, j].set_xlabel("$z$", size=20)
                ax[i, j].set_xlim(z_min, z_max)
                ax[i, j].axvspan(data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om][0], data["x"][0].reshape(hs.bundle_plot_dimension_sizes)[:, 0, Om][int(hs.bundle_plot_dimension_sizes[0]/2)], alpha=0.1, color='grey', label='Training Region')

        handles, labels = ax[-1, -1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.04))
        fig.suptitle("HS Bundle Solutions for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$", size=26, y=.92)
        fig.savefig(f"figures/hs/bundle_examples_hs_{Om}.png", bbox_inches='tight')

def plot_hubble_forward(force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_plot_data(force=force)

    hubble_an = hs.H_HS(data["x"], 1, .3, 65, data["numerical"])
    hubble_fcnn = hs.H_HS(data["x"], 1, .3, 65, data["FCNN"])
    hubble_bbb = hs.H_HS(data["x"], 1, .3, 65, data["BBB_samples"])
    hubble_nlm = hs.H_HS(data["x"], 1, .3, 65, data["NLM_samples"])
    hubble_hmc = hs.H_HS(data["x"], 1, .3, 65, data["HMC_samples"])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(hs.coords_train_min[0], hs.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
    plt.suptitle("HS Hubble Forward", size=26, y=1.05)
    plt.savefig(f"figures/hs/hubble_hs{nan_text}.png", bbox_inches='tight')

def plot_hubble_best_fit(force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_plot_data(force=force)

    b_fcnn, Om_m_0_fcnn, H_0_fcnn = np.load("checkpoints/inverse_samples_bundle_hs_fcnn_cc.npy").mean(axis=0)
    b_bbb, Om_m_0_bbb, H_0_bbb = np.load("checkpoints/inverse_samples_bundle_hs_bbb_cc.npy").mean(axis=0)
    b_nlm, Om_m_0_nlm, H_0_nlm = np.load("checkpoints/inverse_samples_bundle_hs_nlm_cc.npy").mean(axis=0)
    b_hmc, Om_m_0_hmc, H_0_hmc = np.load("checkpoints/inverse_samples_bundle_hs_hmc_cc.npy").mean(axis=0)

    hubble_fcnn = hs.H_HS(data["x"], b_fcnn, Om_m_0_fcnn, H_0_fcnn, data["FCNN"])
    hubble_bbb = hs.H_HS(data["x"], b_bbb, Om_m_0_bbb, H_0_bbb, data["BBB_samples"])
    hubble_nlm = hs.H_HS(data["x"], b_nlm, Om_m_0_nlm, H_0_nlm, data["NLM_samples"])
    hubble_hmc = hs.H_HS(data["x"], b_hmc, Om_m_0_hmc, H_0_hmc, data["HMC_samples"])

    stats = {}
    stats["BBB"] = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    stats["NLM"] = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    stats["HMC"] = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True, dpi=125)

    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        axes[i].axvspan(hs.coords_train_min[0], hs.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
    plt.suptitle("HS Hubble Best Fit", y=1.05, size=26)
    plt.savefig(f"figures/hs/hubble_hs_best_fit{nan_text}.png", bbox_inches='tight')

def plot_hubble_bundle_errors(method, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_bundle_plot_data(force=force)

    z, b, Om_m_0 = data["x"][0].reshape(hs.bundle_plot_dimension_sizes), data["x"][1].reshape(hs.bundle_plot_dimension_sizes), data["x"][2].reshape(hs.bundle_plot_dimension_sizes)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    b_min, b_max = data["x"][1].min(), data["x"][1].max()

    hubble_an = hs.H_HS(z, b, Om_m_0, 65, data["numerical"])
    hubble_fcnn = hs.H_HS(z, b, Om_m_0, 65, data["FCNN"])
    hubble_bbb = hs.H_HS(z, b, Om_m_0, 65, [data["BBB_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_nlm = hs.H_HS(z, b, Om_m_0, 65, [data["NLM_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_hmc = hs.H_HS(z, b, Om_m_0, 65, [data["HMC_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])

    hubble_bbb_mean = np.__dict__[mean_fn](hubble_bbb, axis=0)
    hubble_nlm_mean = np.__dict__[mean_fn](hubble_nlm, axis=0)
    hubble_hmc_mean = np.__dict__[mean_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    fcnn_errors = error_fn(hubble_fcnn, None, hubble_an)
    bbb_errors = error_fn(hubble_bbb_mean, None, hubble_an)
    nlm_errors = error_fn(hubble_nlm_mean, None, hubble_an)
    hmc_errors = error_fn(hubble_hmc_mean, None, hubble_an)

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))
        im1 = ax[0].imshow(fcnn_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")

        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        fig.colorbar(im4, ax=ax[3])

        for j, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[j].set_ylabel("$H_{HS}$", size=20)
            ax[j].set_title(m, size=20)    
            ax[j].set_xlabel("$z$", size=20)

        fig.suptitle("HS Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=1.05)
        fig.savefig(f"figures/hs/hubble_bundle_error_hs{nan_text}_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 4, figsize=(24, 4))
        im1 = ax[0].imshow(fcnn_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
        im2 = ax[1].imshow(nlm_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
        im3 = ax[2].imshow(bbb_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
        im4 = ax[3].imshow(hmc_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")

        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        fig.colorbar(im4, ax=ax[3])

        for j, m in enumerate(["FCNN", "NLM", "BBB", "HMC"]):
            ax[j].set_ylabel("$H_{HS}$", size=20)
            ax[j].set_xlabel("$z$", size=20)
            ax[j].set_title(m, size=20)

        fig.suptitle("HS Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=1.05)
        fig.savefig(f"figures/hs/hubble_bundle_error_hs{nan_text}_{method}_train_{Om}.png", bbox_inches='tight')

def plot_hubble_bundle_std_errors(method, force=False, ignore_nans=False):
    mean_fn = "nanmean" if ignore_nans else "mean"
    std_fn = "nanstd" if ignore_nans else "std"
    nan_text = "_nonan" if ignore_nans else ""
    data = get_bundle_plot_data(force=force)

    z, b, Om_m_0 = data["x"][0].reshape(hs.bundle_plot_dimension_sizes), data["x"][1].reshape(hs.bundle_plot_dimension_sizes), data["x"][2].reshape(hs.bundle_plot_dimension_sizes)
    z_min, z_max = data["x"][0].min(), data["x"][0].max()
    b_min, b_max = data["x"][1].min(), data["x"][1].max()

    hubble_an = hs.H_HS(z, b, Om_m_0, 65, data["numerical"])
    hubble_bbb = hs.H_HS(z, b, Om_m_0, 65, [data["BBB_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_nlm = hs.H_HS(z, b, Om_m_0, 65, [data["NLM_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_hmc = hs.H_HS(z, b, Om_m_0, 65, [data["HMC_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])

    hubble_bbb_mean, hubble_bbb_std = np.__dict__[mean_fn](hubble_bbb, axis=0), np.__dict__[std_fn](hubble_bbb, axis=0)
    hubble_nlm_mean, hubble_nlm_std = np.__dict__[mean_fn](hubble_nlm, axis=0), np.__dict__[std_fn](hubble_nlm, axis=0)
    hubble_hmc_mean, hubble_hmc_std = np.__dict__[mean_fn](hubble_hmc, axis=0), np.__dict__[std_fn](hubble_hmc, axis=0)

    error_fn = error_metrics[method]
    bbb_errors = error_fn(hubble_bbb_mean, hubble_bbb_std, hubble_an).reshape(hs.bundle_plot_dimension_sizes)
    nlm_errors = error_fn(hubble_nlm_mean, hubble_nlm_std, hubble_an).reshape(hs.bundle_plot_dimension_sizes)
    hmc_errors = error_fn(hubble_hmc_mean, hubble_hmc_std, hubble_an).reshape(hs.bundle_plot_dimension_sizes)

    for Om in np.linspace(0, 3, 4):
        Om = int(Om)

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        im2 = ax[0].imshow(nlm_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, :, Om].T, cmap="viridis", origin="lower", extent=[z_min, z_max, b_min, b_max], aspect="auto")

        fig.colorbar(im2, ax=ax[0])
        fig.colorbar(im3, ax=ax[1])
        fig.colorbar(im4, ax=ax[2])

        for j, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[j].set_ylabel("$H_{HS}$", size=20)
            ax[j].set_title(m, size=20)    
            ax[j].set_xlabel("$z$", size=20)

        fig.suptitle("HS Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Test Region)", size=26, y=1.05)
        fig.savefig(f"figures/hs/hubble_bundle_std_error_hs{nan_text}_{method}_test_{Om}.png", bbox_inches='tight')

        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        im2 = ax[0].imshow(nlm_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
        im3 = ax[1].imshow(bbb_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")
        im4 = ax[2].imshow(hmc_errors[:, :, Om].T[:int(hs.bundle_plot_dimension_sizes[0]/2), :], cmap="viridis", origin="lower", extent=[z_min, z_max/2, b_min, b_max], aspect="auto")

        fig.colorbar(im2, ax=ax[0])
        fig.colorbar(im3, ax=ax[1])
        fig.colorbar(im4, ax=ax[2])

        for j, m in enumerate(["NLM", "BBB", "HMC"]):
            ax[j].set_ylabel("$H_{HS}$", size=20)
            ax[j].set_xlabel("$z$", size=20)
            ax[j].set_title(m, size=20)

        fig.suptitle("HS Hubble " + error_names[method] + " for $\Omega_{m,0}" + f" = {round(data['x'][2].reshape(hs.bundle_plot_dimension_sizes)[0, 0, Om], 2)}$ (Train Region)", size=26, y=1.05)
        fig.savefig(f"figures/hs/hubble_bundle_std_error_hs{nan_text}_{method}_train_{Om}.png", bbox_inches='tight')

def plot_calibration(data, name):
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    for i in range(5):
        uct.plot_calibration(data["BBB"][i][0].ravel(), data["BBB"][i][1].ravel(), data["numerical"][i].ravel(), ax=axes[i, 0])
        uct.plot_calibration(data["NLM"][i][0].numpy().ravel(), data["NLM"][i][1].numpy().ravel(), data["numerical"][i].ravel(), ax=axes[i, 1])
        uct.plot_calibration(data["HMC"][i][0].ravel(), data["HMC"][i][1].ravel(), data["numerical"][i].ravel(), ax=axes[i, 2])
        axes[i, 0].set_title("BBB", size=26)
        axes[i, 1].set_title("NLM", size=26)
        axes[i, 2].set_title("HMC", size=26)

    fig.suptitle("HS Forward Solution Calibration", size=26, y=1.05)
    plt.savefig(f"figures/hs/{name}.png", bbox_inches='tight')

def plot_calibration_dts(data_train, data_test, data_ood, name):
    method = "Bundle" if "bundle" in name else "Forward"
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    labels = ["Training Domain", "Testing Domain", "OOD Domain"]
    for i in range(5):
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
    for i in range(4):
        axes[i, 0].set_xlabel("")
        axes[i, 1].set_xlabel("")
        axes[i, 2].set_xlabel("")
    for i in range(1, 5):
        axes[i, 0].set_title("")
        axes[i, 1].set_title("")
        axes[i, 2].set_title("")

    axes[0, 0].annotate("BBB", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[0, 1].annotate("NLM", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
    axes[0, 2].annotate("HMC", size=26, xy=(0.5, 1), xytext=(0, 20), xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')

    for ax, lab in zip(axes[:, 0], ["$x$", "$y$", "$v$", "$\Omega$", "$r$"]):
        ax.annotate(lab, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0), xycoords=ax.yaxis.label, textcoords='offset points', size=26, ha='right', va='center')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, .05))
    fig.suptitle(f"HS {method} Solution Calibration", size=26, y=.93)
    plt.savefig(f"figures/hs/{name}.png", bbox_inches='tight')

if __name__ == "__main__":
    # plot_hs()
    # plot_hubble_forward()
    # plot_hubble_forward(ignore_nans=True)
    # plot_hubble_best_fit()
    # plot_hubble_best_fit(ignore_nans=True)
    # plot_hs_bundle_errors(method="ae")
    # plot_hs_bundle_errors(method="re")
    # plot_hs_bundle_errors(method="rpd")
    # plot_hs_bundle_std_errors(method="ae")
    # plot_hs_bundle_std_errors(method="re")
    # plot_hs_bundle_std_errors(method="rpd")
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
    #     plot_calibration(get_plot_data(domain_type=dt), f"calibration_hs_forward_{dt}")
    #     plot_calibration(get_bundle_plot_data(domain_type=dt), f"calibration_hs_bundle_{dt}")

    plot_calibration_dts(*[get_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_hs_forward_all")
    plot_calibration_dts(*[get_bundle_plot_data(domain_type=dt) for dt in ["train", "test", "ood"]], "calibration_hs_bundle_all")