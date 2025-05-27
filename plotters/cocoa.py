import torch
import dill
import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from neurodiffeq.solvers import Solver1D
import equations.cocoa as cocoa
from models.nlm import NLMModel
from plotters.utils import dill_dec, dill_dec_old
from .config import *

def numerical_cocoa(t):
    max_values = [51.963, 57.741, 13.964053190993093, 7.169932009351017, 5.156121981086967, 0.7365356560864942, 0.4340027187424627, 0.0011931894047004304]
    T = 144
    def func(t, Y):
        x, y, z, u, v, w, r, s = [var * max_values[i] for i, var in enumerate(Y)]
        
        return [
            (T/max_values[0]) * (-cocoa.Y_Glc_Y * cocoa.mu_Y_Glc_max * x * w / (x + cocoa.K_Y_Glc) - cocoa.Y_Glc_LAB * cocoa.mu_LAB_max * x * r / (x + cocoa.K_LAB_Glc)),
            (T/max_values[1]) * (-cocoa.Y_Fru_Y * cocoa.mu_Y_Fru_max * y * w / (y + cocoa.K_Y_Fru)),
            (T/max_values[2]) * (cocoa.Y_Glc_EtOH_Y * cocoa.mu_Y_Glc_max * x * w / (x + cocoa.K_Y_Glc) + cocoa.Y_Fru_EtOH_Y * cocoa.mu_Y_Fru_max *y * w / (y + cocoa.K_Y_Fru) - cocoa.Y_EtOH_AAB * cocoa.mu_AAB_EtOH_max * z * s / (z + cocoa.K_AAB_EtOH)),
            (T/max_values[3]) * (cocoa.Y_LA_LAB * cocoa.mu_LAB_max * x * r / (x + cocoa.K_LAB_Glc) - cocoa.Y_LA_AAB * cocoa.mu_AAB_LA_max * u * s / (u + cocoa.K_AAB_LA * s)),
            (T/max_values[4]) * (cocoa.Y_Ac_LAB * cocoa.mu_LAB_max * x * r / (x + cocoa.K_LAB_Glc) + cocoa.Y_EtOH_Ac_AAB * cocoa.mu_AAB_EtOH_max * z * s / (z + cocoa.K_AAB_EtOH) + cocoa.Y_LA_Ac_AAB * cocoa.mu_AAB_LA_max * u * s / (u + cocoa.K_AAB_LA * s)),
            (T/max_values[5]) * (cocoa.mu_Y_Glc_max * x * w / (x + cocoa.K_Y_Glc) + cocoa.mu_Y_Fru_max * y * w / (y + cocoa.K_Y_Fru) - cocoa.k_Y * w * z),
            (T/max_values[6]) * (cocoa.mu_LAB_max * x * r / (x + cocoa.K_LAB_Glc) - cocoa.k_LAB * r * u),
            (T/max_values[7]) * (cocoa.mu_AAB_EtOH_max * z * s / (z + cocoa.K_AAB_EtOH) + cocoa.mu_AAB_LA_max * u * s / (u + cocoa.K_AAB_LA * s) - cocoa.k_AAB * s * v**2)
        ]
        
    initial_conditions = np.array([51.963, 57.741, 0, 0, 0, 0.029180401, 0.007868827, 3.36634e-6])
    initial_conditions = initial_conditions / np.array(max_values)

    rk4_sol = RK45(func, t0=cocoa.coords_test_min[0], y0=initial_conditions, t_bound=cocoa.coords_test_max[0], max_step=0.001)

    t_values = [cocoa.coords_test_min[0]]
    x_values = [initial_conditions[0]]
    y_values = [initial_conditions[1]]
    z_values = [initial_conditions[2]]
    u_values = [initial_conditions[3]]
    v_values = [initial_conditions[4]]
    w_values = [initial_conditions[5]]
    r_values = [initial_conditions[6]]
    s_values = [initial_conditions[7]]

    while rk4_sol.status != "finished":
        rk4_sol.step()
        
        t_values.append(rk4_sol.t)
        x_values.append(rk4_sol.y[0])
        y_values.append(rk4_sol.y[1])
        z_values.append(rk4_sol.y[2])
        u_values.append(rk4_sol.y[3])
        v_values.append(rk4_sol.y[4])
        w_values.append(rk4_sol.y[5])
        r_values.append(rk4_sol.y[6])
        s_values.append(rk4_sol.y[7])

    rk4_t = np.array(t_values)
    rk4_x_points = np.array(x_values)
    rk4_y_points = np.array(y_values)
    rk4_z_points = np.array(z_values)
    rk4_u_points = np.array(u_values)
    rk4_v_points = np.array(v_values)
    rk4_w_points = np.array(w_values)
    rk4_r_points = np.array(r_values)
    rk4_s_points = np.array(s_values)
    x = np.interp(t, rk4_t, rk4_x_points)
    y = np.interp(t, rk4_t, rk4_y_points)
    z = np.interp(t, rk4_t, rk4_z_points)
    u = np.interp(t, rk4_t, rk4_u_points)
    v = np.interp(t, rk4_t, rk4_v_points)
    w = np.interp(t, rk4_t, rk4_w_points)
    r = np.interp(t, rk4_t, rk4_r_points)
    s = np.interp(t, rk4_t, rk4_s_points)
    return x, y, z, u, v, w, r, s

@dill_dec_old("plot_data/cocoa_forward.dill")
def get_plot_data():
    x_test = np.linspace(cocoa.coords_test_min, cocoa.coords_test_max, 200).reshape(-1, 1)
    numerical = numerical_cocoa(x_test)

    solver = Solver1D.load("checkpoints/solver_cocoa_fcnn.ndeq")
    solution = solver.get_solution()(x_test, to_numpy=True)

    solver_bbb = torch.load(f"checkpoints/solver_cocoa_bbb.pyro")
    bbb_samples = solver_bbb.posterior_predictive([torch.tensor(x_test)], num_samples=10_000)

    nlm_model = NLMModel.load(f"checkpoints/model_cocoa_nlm.pt")
    nlm_model.device = "cpu"
    nlm_model.get_likelihood_stds.device = "cpu"
    nlm_means, nlm_stds = nlm_model.posterior_predictive([x_test], include_Sigmas_e=True)

    hmc_solver = torch.load(f"checkpoints/solver_cocoa_hmc.pyro", pickle_module=dill)
    hmc_posterior_samples = torch.load(f"checkpoints/samples_cocoa_hmc.pyro", pickle_module=dill)
    hmc_samples = hmc_solver.posterior_predictive([torch.tensor(x_test)], hmc_posterior_samples, to_numpy=True)

    data = {
         "x": x_test,
         "numerical": numerical,
         "FCNN": solution,
         "BBB": [(bbb_samples[i].mean(axis=0), bbb_samples[i].std(axis=0)) for i in range(8)],
         "NLM": [(nlm_means[i].detach().cpu(), nlm_stds[i].detach().cpu()) for i in range(8)],
         "HMC": [(hmc_samples[i].mean(axis=0), hmc_samples[i].std(axis=0)) for i in range(8)],
         }
    return data

def plot_cocoa():
    data = get_plot_data()
    func_names = ["$x(t)$", "$y(t)$", "$z(t)$", "$u(t)$", "$v(t)$", "$w(t)$", "$r(t)$", "$s(t)$"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharey="row", sharex=True, dpi=125)

    i = 0
    for i, method in enumerate(["BBB", "NLM", "HMC"]):
        for j in range(8):
            axes[0][i].axvspan(cocoa.coords_test_min[0], cocoa.coords_train_max[0], alpha=0.1, color='grey', label='Training Region')
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
    #plt.savefig("figures/cocoa.png", bbox_inches="tight")
    plt.show()