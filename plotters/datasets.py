import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def load_cc(filename):
    df = pd.read_csv(filename)
    df.sort_values(by="z", inplace=True, ascending=True)
    return df["z"].values, df["h"].values, df["std"].values

def load_lab(filename):
    df = pd.read_csv(filename)
    df.sort_values(by="t", inplace=True, ascending=True)
    return df["t"].values, df["x"].values, df["y"].values, df["z"].values

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
plt.rc('font', family='serif', size=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=12)

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ## Plotting the cosmic chronometers data
    cc_z, cc_h, cc_std = load_cc("datasets/cc.csv")
    lcdm = load_cc("datasets/cc_lcdm_syn.csv")
    cpl = load_cc("datasets/cc_cpl_syn.csv")
    quint = load_cc("datasets/cc_quint_syn.csv")
    hs = load_cc("datasets/cc_hs_syn.csv")

    plt.plot(lcdm[0], lcdm[1], ".-", label='LCDM')
    plt.plot(cpl[0], cpl[1], ".-", label='CPL')
    plt.plot(quint[0], quint[1], ".-", label='Quintessence')
    plt.plot(hs[0], hs[1], ".-", label='HS')
    plt.plot(cc_z, cc_h, ".", label='CC', color='black', alpha=.5)
    plt.errorbar(cc_z, cc_h, yerr=cc_std, fmt='none', ecolor='black', alpha=.5, elinewidth=1, capsize=2)
    plt.xlabel('$z$')
    plt.ylabel('$H(z)$')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('figures/hubble_syn.png')

    ## Plotting the lab data
    lab = load_lab("datasets/lab.csv")
    lab_syn = load_lab("datasets/lab_syn.csv")


    axes[0].set_ylabel("$x(t)$")
    axes[0].set_xlabel("$t$")
    axes[0].plot(lab_syn[0], lab_syn[1], ".-", label="Synthetic")
    axes[0].plot(lab[0], lab[1], ".-", label="Observed")
    axes[0].legend(loc="best")

    axes[1].set_ylabel("$y(t)$")
    axes[1].set_xlabel("$t$")
    axes[1].plot(lab_syn[0], lab_syn[2], ".-", label="Synthetic")
    axes[1].plot(lab[0], lab[2], ".-", label="Observed")
    axes[1].legend(loc="best")

    axes[2].set_ylabel("$z(t)$")
    axes[2].set_xlabel("$t$")
    axes[2].plot(lab_syn[0], lab_syn[3], ".-", label="Synthetic")
    axes[2].plot(lab[0], lab[3], ".-", label="Observed")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig("figures/lab_syn.png")