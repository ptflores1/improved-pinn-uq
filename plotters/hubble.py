import matplotlib.pyplot as plt

from .datasets import load_cc
from . import lcdm
from . import cpl
from . import quintessence
from . import hs

cc_z, cc_h, cc_std = load_cc("datasets/cc.csv")

fig, ax = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

lcdm_data = lcdm.get_plot_data()
lcdm_data_eb = lcdm.get_plot_data(True)
cpl_data = cpl.get_plot_data()
cpl_data_eb = cpl.get_plot_data(True)
quintessence_data = quintessence.get_plot_data()
hs_data = hs.get_plot_data()

for i, method in enumerate(["fcnn", "nlm", "bbb", "hmc"]):
    ax[i].plot(cc_z, cc_h, ".", label="Cosmic Chronometers", color='black', alpha=.5)
    ax[i].errorbar(cc_z, cc_h, yerr=cc_std, fmt='none', ecolor='black', elinewidth=1, capsize=2, alpha=.5)
    ax[i].set_xlabel("$z$")
    if i == 0: ax[i].set_ylabel("$H(z)$")
    ax[i].set_title(method.upper())
    for equation in ["lcdm", "cpl", "quintessence", "hs"]:
        ax[i].

handles, labels = ax[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()