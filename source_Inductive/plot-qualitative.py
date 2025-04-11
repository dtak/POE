import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

externals = [0., 0.0001, 0.001, 0.01, 0.1]
internals = [0., 0.0001, 0.01, 10., 1000.]

for sample in [0, 2]:

    res = [[None for _ in internals] for _ in externals]
    for e, external in enumerate(externals):
        for i, internal in enumerate(internals):
            res[e][i] = Image.open(f"results/perturbations-sample{sample}/external({external})-internal({internal}).png")
            res[e][i] = np.array(res[e][i])

    fig, axs = plt.subplots(
        nrows = len(externals),
        ncols = len(internals),
        figsize = (.5 + len(internals) * 244 / 200, .5 + len(externals) * 244 / 200),
        dpi = 500)

    for e, external in enumerate(externals):
        for i, internal in enumerate(internals):
            _e = len(externals) - e - 1
            axs[_e,i].imshow(res[e][i], interpolation='nearest')
            axs[_e,i].set_xticks([])
            axs[_e,i].set_yticks([])
            if e == 0: axs[_e,i].set_xlabel(internal)
            if i == 0: axs[_e,i].set_ylabel(external)

    fig.supxlabel('Smoothness Parameter ($\\lambda_{\\text{smooth}}$)')
    fig.supylabel('Robustness Parameter ($\\lambda_{\\text{robust}}$)')
    fig.suptitle("Our Approach", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"results/sample{sample}.pdf")
