import os
from getdist import plots, MCSamples
import matplotlib
import matplotlib.pyplot as plt
from .config import *

def plot_inverse_samples(param_samples, param_names, param_labels, plot_label, name):
    samples = MCSamples(samples=param_samples, names=param_names, labels=param_labels, label=plot_label)

    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True, legend_labels=[plot_label], legend_loc='upper right')
    g.export(f'figures/inverse_samples_{name}.png')

if __name__ == "__main__":
    import numpy as np

    equation_params_n_labels = {
        "lcdm": (["Om_m_0", "H_0"], ["$\Omega_{m_0}$", "$H_0$"]),
        "cpl": (["w0", "w1", "Om_m_0", "H_0"], ["$w_0$", "$w_1$", "$\Omega_{m_0}$", "$H_0$"]),
        "quintessence": (["lambda", "Om_m_0", "H_0"], ["$\lambda$", "$\Omega_{m_0}$", "$H_0$"]),
        "hs": (["b", "Om_m_0", "H_0"], ["$b$", "$\Omega_{m_0}$", "$H_0$"]),
        "lab": (["alpha", "delta", "sigma"], ["$\\alpha$", "$\delta$", "$\sigma$"])
    }

    for f in os.listdir("checkpoints"):
        if f.startswith("inverse_samples"):
            samples = np.load(f"checkpoints/{f}")
            experiment = f.split(".")[0].split("_")[3:]
            plot_inverse_samples(samples, *equation_params_n_labels[experiment[0]], name="_".join(experiment), plot_label="Cosmic Chronometers")