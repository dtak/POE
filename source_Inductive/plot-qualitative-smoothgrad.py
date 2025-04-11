import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

deltas = [0., 0.01, 0.05, 0.25, 1.25]

for sample in [0, 2]:

    res = [None for _ in deltas]
    for i, delta in enumerate(deltas):
        
        if i == 0:
            res[0] = Image.open(f"results/perturbations-sample{sample}/external(0.0)-internal(0.0).png")
            res[0] = np.array(res[i])
            continue

        res[i] = Image.open(f"results/perturbations-sample{sample}-delta({delta})/smoothgrad.png")
        res[i] = np.array(res[i])
        
    fig, axs = plt.subplots(
        nrows = len(deltas),
        ncols = 1,
        figsize = (.5 + 244 / 200, .5 + len(deltas) * 244 / 200),
        dpi = 500)

    for i, delta in enumerate(deltas):
        _i = len(deltas) - i -1
        axs[_i].imshow(res[i], interpolation='nearest')
        axs[_i].set_xticks([])
        axs[_i].set_yticks([])
        axs[_i].set_ylabel(int(delta * 100))
        if i == 0: axs[_i].set_xlabel(' ')

    fig.supxlabel(' ')
    fig.supylabel('Noise Level (%)')
    fig.suptitle("SmoothGrad", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"results/smoothgrad-sample{sample}.pdf")
