import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
# import seaborn as sns

N_SEEDS = 5

# sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(4 * 3, 3 * 2))

function_classes = [
    "functions/cubed-seed{seed}-lambda({_lambda}).csv",
    "functions/exp_quasi-seed{seed}-lambda({_lambda}).csv",
    "functions/exponential-seed{seed}-lambda({_lambda}).csv",
    "functions/sine-seed{seed}-lambda({_lambda}).csv",
    "functions/quasi-seed{seed}-lambda({_lambda}).csv",
    "imagenet2000-seed{seed}/inductive-lambda({_lambda}).csv"]

for ax, function_class, title in zip(
    axes.ravel(),
    function_classes, [
        "Power function: cubic", "Quasi-periodic w/ exponent", "Exponential function",
        "Periodic function: sine", "Quasi-periodic w/ linear term", "CNN: ImageNet-1k"]):

    for _lambda, color in zip(
        [0.01, 0.1],
        ['tab:orange', 'tab:red']):

        df0 = pd.read_csv("results/" + function_class.format(seed=0, _lambda=_lambda), index_col=0)

        ns = df0.index.values
        res_F = np.zeros((N_SEEDS, df0.index.size))
        res_R = np.zeros((N_SEEDS, df0.index.size))
        res_F0 = np.zeros((N_SEEDS, df0.index.size))
        res_R0 = np.zeros((N_SEEDS, df0.index.size))

        for seed in range(N_SEEDS):
            df = pd.read_csv("results/" + function_class.format(seed=seed, _lambda=_lambda), index_col=0)
            
            res_F[seed] = df.F
            res_R[seed] = df.R
            res_F0[seed] = df.F0
            res_R0[seed] = df.R0

        res_L = res_F + _lambda * res_R
        res_L0 = res_F0 + _lambda * res_R0

        ax.plot(ns, res_L.mean(0), color=color)
        ax.fill_between(ns, res_L.mean(0)-res_L.std(0), res_L.mean(0)+res_L.std(0), color=color, alpha=.2)

        ax.plot(ns, res_L0.mean(0), color=color, linestyle="--")
        ax.fill_between(ns, res_L0.mean(0)-res_L0.std(0), res_L0.mean(0)+res_L0.std(0), color=color, alpha=.2)

    ax.set_xscale('log')
    ax.set_xlabel("Number of Inducing Points ($N$)")
    ax.set_ylabel("Loss (Total)")

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.set_title(title) #, fontweight='bold')
    
    legend_lines = [
        Line2D([0], [0], color="black"),
        Line2D([0], [0], color="black", linestyle='--'),
        Line2D([0], [0], color="tab:orange"),
        Line2D([0], [0], color="tab:red")]
    ax.legend(legend_lines, [
        "Inductive",
        "Transductive",
        "$\\lambda_{\\text{robust}}=0.01$",
        "$\\lambda_{\\text{robust}}=0.1$"]) 
###

plt.tight_layout()
plt.savefig("results/inductive-master.pdf")
