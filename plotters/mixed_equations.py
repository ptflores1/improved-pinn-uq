import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotters.lcdm as plcdm
import plotters.cpl as pcpl
import plotters.quintessence as pquint
import plotters.hs as phs
from plotters.datasets import load_cc

def plot_best_fit():
    lcdm = plcdm.get_best_fit_plot_data()
    lcdm_eb = plcdm.get_best_fit_plot_data(eb=True)
    cpl = pcpl.get_best_fit_plot_data()
    cpl_eb = pcpl.get_best_fit_plot_data(eb=True)
    quint = pquint.get_best_fit_plot_data()
    hs = phs.get_best_fit_plot_data()
    cc_z, cc_h, cc_std = load_cc("datasets/cc.csv")

    eqs_data = [lcdm, lcdm_eb, cpl, cpl_eb, quint, hs]
    eq_names = ["$\Lambda$CDM", "$\Lambda$CDM+EB", "CPL", "CPL+EB", "Quintessence", "HS"]
    colors = list(mcolors.TABLEAU_COLORS.keys())[1:]

    fig, axes = plt.subplots(1, 4, figsize=(24, 4), sharex=True)
    for eqi, (ed, eq_name) in enumerate(zip(eqs_data, eq_names)):
        axes[0].set_title("FCNN")
        if not "EB" in eq_name: axes[0].plot(ed["x"], ed["FCNN"], colors[eqi], label=eq_name)
        axes[0].set_xlabel("$z$", size=26)
        for i, method in enumerate(["BBB", "NLM", "HMC"]):
            axes[i+1].set_title(method)
            axes[i+1].set_xlabel("$z$", size=26)
            axes[i+1].plot(ed["x"], ed[method][0], colors[eqi], linewidth=2, label=eq_name)

    for i in range(4):
        axes[i].errorbar(cc_z, cc_h, yerr=cc_std, color="tab:blue", fmt='o', markersize=3, label="CC")
    
    axes[0].set_ylabel("$H(z)$", size=26)
    axes[-1].set_xlim(0, 2)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.22))
    plt.savefig(f"figures/hubble_best_fits.png", bbox_inches='tight')

if __name__ == "__main__":
    plot_best_fit()