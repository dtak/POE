import argparse
import matplotlib.image as image
import pathlib
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

###

raw_x, raw_y = torch.load(f"datasets/{args.dataset}.pt", weights_only=False)

x = raw_x.mean(dim=1)
y = raw_y.mean(dim=1)
y = torch.maximum(y, torch.zeros_like(y))
y = torch.minimum(y, torch.quantile(y.reshape(y.shape[0], -1), q=0.99, dim=-1)[:,None,None])
y = y - y.reshape(y.shape[0], -1).min(-1).values[:,None,None]
y = y / y.reshape(y.shape[0], -1).max(-1).values[:,None,None]

###

res = torch.mean(y, dim=0)
pathlib.Path(f"results/{args.dataset}").mkdir(parents=True, exist_ok=True)
image.imsave(f"results/{args.dataset}/smoothgrad.png", res, cmap='binary_r')
