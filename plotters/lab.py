import torch
import dill
import numpy as np
from scipy.integrate import RK45
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from neurodiffeq.solvers import Solver1D, BundleSolver1D
import equations.lab as lab
from models.nlm import NLMModel
from plotters.utils import dill_dec, dill_dec_old
from .config import *

def numerical_lab(t, alpha=lab.alpha, delta=lab.delta, sigma=lab.sigma):
    def func(t, Y):
        x, y, z = Y
        
        return [
            alpha*x - lab.beta*x**2 - lab.gamma*x*y**2 - lab.tau*x*z**2,
            delta*y - lab.theta*y**2 - lab.rho*y*z,
            sigma*z - lab.phi*z**2 - lab.omega*x*z,
        ]
        
    initial_conditions = np.array([7.5797, 6.44, 1.9])
    rk4_sol = RK45(func, t0=lab.coords_test_min[0], y0=initial_conditions, t_bound=lab.coords_test_max[0], max_step=0.01)

    t_values = [lab.coords_test_min[0]]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]
    z_values = [initial_conditions[2]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])
        z_values.append(rk4_sol.y[2])

    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)
    rk4_z_points = np.array(z_values)
    x = np.interp(t, rk4_t, rk4_x_points)
    y = np.interp(t, rk4_t, rk4_y_points)
    z = np.interp(t, rk4_t, rk4_z_points)
    return x, y, z

def numerical_lab_bundle(t, alpha, delta, sigma):
    sol_x = np.zeros((t.shape[0], alpha.shape[0], delta.shape[0], sigma.shape[0]))
    sol_y = np.zeros((t.shape[0], alpha.shape[0], delta.shape[0], sigma.shape[0]))
    sol_z = np.zeros((t.shape[0], alpha.shape[0], delta.shape[0], sigma.shape[0]))
    for i, a in tqdm(enumerate(alpha.ravel()), desc="Bundle Numerical alpha", total=alpha.shape[0]):
        for j, d in tqdm(enumerate(delta.ravel()), leave=False, desc="Bundle Numerical delta", total=delta.shape[0]):
            for k, s in tqdm(enumerate(sigma.ravel()), leave=False, desc="Bundle Numerical sigma", total=sigma.shape[0]):
                x, y, z = numerical_lab(t, a, d, s)
                sol_x[:, i, j, k] = x.ravel()
                sol_y[:, i, j, k] = y.ravel()
                sol_z[:, i, j, k] = z.ravel()
    return sol_x, sol_y, sol_z

@dill_dec("lab")
def get_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        x_test = np.linspace(lab.coords_test_min, lab.coords_test_max, 200).reshape(-1, 1)
    elif domain_type == "train":
        x_test = np.linspace(lab.coords_train_min, lab.coords_train_max, 200).reshape(-1, 1)
    else:
        x_test = np.linspace(lab.coords_train_max, lab.coords_test_max, 200).reshape(-1, 1)
    numerical = numerical_lab(x_test)

    batch_size = 10
    batches = 10_000 // batch_size

    solver = Solver1D.load("checkpoints/solver_lab_fcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(x_test, to_numpy=True)

    solver_bbb = torch.load(f"checkpoints/solver_lab_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = lab.system
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x_test)], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp[0])
        bbb_residuals.append(bbb_residuals_tmp[0])
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(3)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(3)]

    nlm_model = NLMModel.load(f"checkpoints/model_lab_nlm.pt")
    nlm_model.diff_eqs = lab.system
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_means, nlm_stds = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive([x_test], n_samples=10_000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals([x_test])

    hmc_solver = torch.load(f"checkpoints/solver_lab_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = lab.system
    hmc_solver.get_likelihood_std.device = "cpu"
    hmc_posterior_samples = torch.load(f"checkpoints/samples_lab_hmc.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x_test)], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp[0])
        hmc_residuals.append(hmc_residuals_tmp[0])
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(3)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(3)]

    data = {
        "domain_type": domain_type,
         "x": x_test,
         "numerical": numerical,
         "FCNN": solution,
         "BBB": [(bbb_samples[i].mean(axis=0), bbb_samples[i].std(axis=0)) for i in range(3)],
         "NLM": [(nlm_means[i].detach().cpu(), nlm_stds[i].detach().cpu()) for i in range(3)],
         "HMC": [(hmc_samples[i].mean(axis=0), hmc_samples[i].std(axis=0)) for i in range(3)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": [r.detach() for r in nlm_residuals],
         "HMC_residuals": hmc_residuals,
         }
    return data

@dill_dec("lab", bundle=True)
def get_bundle_plot_data(eb=False, domain_type="test"):
    if domain_type == "test":
        t_test = np.linspace(lab.coords_test_min, lab.coords_test_max, lab.bundle_plot_dimension_sizes[0])
    elif domain_type == "train":
        t_test = np.linspace(lab.coords_train_min, lab.coords_train_max, lab.bundle_plot_dimension_sizes[0])
    else:
        t_test = np.linspace(lab.coords_train_max, lab.coords_test_max, lab.bundle_plot_dimension_sizes[0])

    params_test = [np.linspace(lab.bundle_parameters_min_plot[i], lab.bundle_parameters_max_plot[i], lab.bundle_plot_dimension_sizes[i+1]) for i in range(len(lab.bundle_parameters_min))]
    x_test = [x.reshape(-1, 1) for x in np.meshgrid(t_test, *params_test, indexing="ij")]
    numerical = numerical_lab_bundle(t_test, *params_test)

    batch_size = 10
    batches = 10000 // batch_size

    solver = BundleSolver1D.load("checkpoints/solver_bundle_lab_fcnn.ndeq")
    solution = solver.get_solution()(*x_test, to_numpy=True)
    fcnn_residuals = solver.get_residuals(*x_test, to_numpy=True)

    solver_bbb = torch.load(f"checkpoints/solver_bundle_lab_bbb.pyro", map_location="cpu")
    solver_bbb.diff_eqs = lab.system_bundle
    if "get_likelihood_std" in solver_bbb.__dict__:
        solver_bbb.get_likelihood_std.device = "cpu"
    bbb_samples = []
    bbb_residuals = []
    for _ in tqdm(range(batches), desc="BBB Samples"):
        bbb_samples_tmp, bbb_residuals_tmp = solver_bbb.posterior_predictive([torch.tensor(x) for x in x_test], num_samples=batch_size, to_numpy=True, include_residuals=True)
        bbb_samples.append(bbb_samples_tmp)
        bbb_residuals.append(bbb_residuals_tmp)
    bbb_samples = [np.concatenate([s[i] for s in bbb_samples], axis=0) for i in range(3)]
    bbb_residuals = [np.concatenate([s[i] for s in bbb_residuals], axis=0) for i in range(3)]

    nlm_model = NLMModel.load(f"checkpoints/model_bundle_lab_nlm.pt")
    nlm_model.diff_eqs = lab.system_bundle
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_mean, nlm_std = nlm_model.posterior_predictive(x_test, include_Sigmas_e=True)
    nlm_samples = nlm_model.sample_posterior_predictive(x_test, n_samples=10000, to_numpy=True)
    nlm_residuals = nlm_model.get_residuals(x_test)
    nlm_mean = [m.detach().cpu() for m in nlm_mean]
    nlm_std = [m.detach().cpu() for m in nlm_std]
    nlm_residuals = [r.detach() for r in nlm_residuals]

    hmc_solver = torch.load(f"checkpoints/solver_bundle_lab_hmc.pyro", pickle_module=dill, map_location="cpu")
    hmc_solver.diff_eqs = lab.system_bundle
    hmc_posterior_samples = torch.load(f"checkpoints/samples_bundle_lab_hmc.pyro", pickle_module=dill, map_location="cpu")
    n_hmc_samples = list(hmc_posterior_samples.values())[0].shape[0]
    batches = n_hmc_samples // batch_size
    hmc_samples = []
    hmc_residuals = []
    for i in tqdm(range(batches), desc="HMC Samples"):
        post_samples = { k: v[i*batch_size:(i+1)*batch_size] for k, v in hmc_posterior_samples.items()}
        hmc_samples_tmp, hmc_residuals_tmp = hmc_solver.posterior_predictive([torch.tensor(x) for x in x_test], post_samples, to_numpy=True, include_residuals=True)
        hmc_samples.append(hmc_samples_tmp)
        hmc_residuals.append(hmc_residuals_tmp)
    hmc_samples = [np.concatenate([s[i] for s in hmc_samples], axis=0) for i in range(3)]
    hmc_residuals = [np.concatenate([s[i] for s in hmc_residuals], axis=0) for i in range(3)]

    data = {
         "domain_type": domain_type,
         "x": x_test,
         "numerical": [num.reshape(lab.bundle_plot_dimension_sizes) for num in numerical],
         "FCNN": solution.reshape(lab.bundle_plot_dimension_sizes),
         "BBB": [(bbb_samples[i].mean(axis=0).reshape(lab.bundle_plot_dimension_sizes), bbb_samples[i].std(axis=0).reshape(lab.bundle_plot_dimension_sizes)) for i in range(3)],
         "NLM": [(nlm_mean[i].detach().cpu().reshape(lab.bundle_plot_dimension_sizes), nlm_std[i].detach().cpu().reshape(lab.bundle_plot_dimension_sizes)) for i in range(3)],
         "HMC": [(hmc_samples[i].mean(axis=0).reshape(lab.bundle_plot_dimension_sizes), hmc_samples[i].std(axis=0).reshape(lab.bundle_plot_dimension_sizes)) for i in range(3)],
         "BBB_samples": bbb_samples,
         "NLM_samples": nlm_samples,
         "HMC_samples": hmc_samples,
         "FCNN_residuals": fcnn_residuals,
         "BBB_residuals": bbb_residuals,
         "NLM_residuals": nlm_residuals,
         "HMC_residuals": hmc_residuals,
         }
    return data

def plot_lab():
    data = get_plot_data()
    func_names = ["$x(t)$", "$y(t)$", "$z(t)$"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey="row", sharex=True, dpi=125)

    i = 0
    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        for j in range(3):
            axes[0][i].axvspan(lab.coords_test_min[0], lab.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
            axes[0][i].plot(data["x"], data["numerical"][j], 'olive', linestyle='--', alpha=0.75, linewidth=2, label='Ground Truth')
            axes[0][i].plot(data["x"], data["FCNN"][j], "red", alpha=.6, label="Det Solution")
            axes[0][i].plot(data["x"], data[method][j][0], 'darkslateblue', linewidth=2, alpha=1, label='Mean of Post. Pred.')
            axes[0][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-1*data[method][j][1].ravel(), data[method][j][0].ravel()+1*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None, label='Uncertainty')
            axes[0][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-2*data[method][j][1].ravel(), data[method][j][0].ravel()+2*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)
            axes[0][i].fill_between(data["x"].ravel(), data[method][j][0].ravel()-3*data[method][j][1].ravel(), data[method][j][0].ravel()+3*data[method][j][1].ravel(), color="darkslateblue", alpha=0.2, edgecolor=None)

            axes[0][i].set_xlim(data["x"].min(), data["x"].max())
            if j == 0: axes[j][i].set_title(method)
            if j == 1: axes[j][i].set_xlabel("$t$", size=26)
            if i == 0: axes[j][i].set_ylabel(func_names[j], size=26)

    handles, labels = axes[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
    #plt.savefig("figures/lab.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    pass
    #data = get_plot_data()
    #print(data)