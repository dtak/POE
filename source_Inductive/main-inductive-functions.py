import argparse
import numpy as np
import pandas as pd
import pathlib
import torch

from inductive.methods import gp_robust
from inductive.functions import create

parser = argparse.ArgumentParser()
parser.add_argument('--function', required=True)
parser.add_argument('--seed', required=True)
args = parser.parse_args()

params = dict()
params['noise'] = 1.
params['lengthscale'] = 1.

###

x, y, x_train, y_train = create(args.function, seed=args.seed)
NS = list(np.logspace(2, 3, 10).round().astype(int))

x_train = x_train.to(dtype=torch.double)
y_train = y_train.to(dtype=torch.double)
x = x.to(dtype=torch.double)
y = y.to(dtype=torch.double)

###

_K = torch.einsum('n...,n...->n', x, x)
_K = _K[:,None] + _K[None,:] - 2 * torch.einsum('n...,N...->nN', x, x)
_K = _K / x.shape[1:].numel() / params['lengthscale']**2

K = torch.exp(-0.5 * _K)
K_INVERSE = torch.linalg.inv(K)
K_INVERSE_ROWSUMS = torch.sum(K_INVERSE, dim=1)

def compute_faithfulness(res):
    return torch.sum((res - y)**2)

def compute_robustness(res):
    norms = torch.einsum('n...,n...->n', res, res)
    diffs = norms[:,None] + norms[None,:] - 2 * torch.einsum('n...,N...->nN', res, res)
    return -0.5 * torch.tensordot(diffs, K_INVERSE), torch.dot(norms, K_INVERSE_ROWSUMS)

path = f"results/functions"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

for _lambda in [0.1, 0.01]:
    params['lambda'] = _lambda

    df = pd.DataFrame(
        index = NS,
        columns = ['F', 'R', 'RR', 'F0', 'R0', 'RR0'])

    for N in NS:
        print(f"running lambda = {_lambda}, N = {N} ...")

        res = gp_robust(x, y, x_train[:N], y_train[:N], params, mode='inductive')

        F = compute_faithfulness(res)
        R, RR = compute_robustness(res)
        df.loc[N].F = F.item()
        df.loc[N].R = R.item()
        df.loc[N].RR = RR.item()

        res = gp_robust(x, y, x_train[:N], y_train[:N], params, mode='transductive')

        F = compute_faithfulness(res)
        R, RR = compute_robustness(res)
        df.loc[N].F0 = F.item()
        df.loc[N].R0 = R.item()
        df.loc[N].RR0 = RR.item()

    df.to_csv(f"results/functions/{args.function}-seed{args.seed}-lambda({_lambda}).csv")
