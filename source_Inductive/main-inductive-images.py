import argparse
import numpy as np
import pandas as pd
import pathlib
import torch

from inductive.methods import gp_robust

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

params = dict()
params['noise'] = 1.
params['lengthscale'] = 10.

N_TEST = 1000
NS = list(np.logspace(1, 3, 20).round().astype(int))

###

raw_x, raw_y = torch.load(f"datasets/{args.dataset}.pt", weights_only=False)
# raw_x = raw_x[:2000]
# raw_y = raw_y[:2000]

x = raw_x.mean(dim=1)
y = raw_y.mean(dim=1)
y = torch.maximum(y, torch.zeros_like(y))
y = torch.minimum(y, torch.quantile(y.reshape(y.shape[0], -1), q=0.99, dim=-1)[:,None,None])
y = y - y.reshape(y.shape[0], -1).min(-1).values[:,None,None]
y = y / y.reshape(y.shape[0], -1).max(-1).values[:,None,None]

x_train, y_train = x[N_TEST:], y[N_TEST:]
x, y = x[:N_TEST], y[:N_TEST]

###

x_train = x_train.to(dtype=torch.double)
y_train = y_train.to(dtype=torch.double)
x = x.to(dtype=torch.double)
y = y.to(dtype=torch.double)

###

_K = torch.einsum('n...,n...->n', x, x)
_K = _K[:,None] + _K[None,:] - 2 * torch.einsum('n...,N...->nN', x, x)
_K = _K / x.shape[1:].numel() / params['lengthscale']**2

K = torch.exp(-0.5 * _K) #/ params['lambda']
K_INVERSE = torch.linalg.inv(K)
K_INVERSE_ROWSUMS = torch.sum(K_INVERSE, dim=1)

def compute_faithfulness(res):
    return torch.sum((res - y)**2)

def compute_robustness(res):
    norms = torch.einsum('n...,n...->n', res, res)
    diffs = norms[:,None] + norms[None,:] - 2 * torch.einsum('n...,N...->nN', res, res)
    return -0.5 * torch.tensordot(diffs, K_INVERSE), torch.dot(norms, K_INVERSE_ROWSUMS)

path = f"results/{args.dataset}"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

for _lambda in [0.001]: #[0.01, 0.1, 1.]:
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

    df.to_csv(f"results/{args.dataset}/inductive-lambda({_lambda}).csv")
