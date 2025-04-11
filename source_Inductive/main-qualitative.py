import argparse
import matplotlib.image as image
import pathlib
import torch

from inductive.methods import gp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('-e', '--external', type=float, default=0.)
parser.add_argument('-i', '--internal', type=float, default=0.)
args = parser.parse_args()

args.dataset_name = args.dataset
args.dataset_size = None
if args.dataset.find('[') != -1:
    args.dataset_name = args.dataset[:args.dataset.find('[')]
    args.dataset_size = int(args.dataset[args.dataset.find('[')+1:-1])

params = dict()
params['noise'] = 1.
params['lengthscale_external'] = 10.
params['lengthscale_internal'] = 0.1
params['lambda_external'] = args.external
params['lambda_internal'] = args.internal
params['double'] = True # double precision

###

raw_x, raw_y = torch.load(f"datasets/{args.dataset_name}.pt", weights_only=False)
raw_x = raw_x[:args.dataset_size]
raw_y = raw_y[:args.dataset_size]

x = raw_x.mean(dim=1)
y = raw_y.mean(dim=1)
y = torch.maximum(y, torch.zeros_like(y))
y = torch.minimum(y, torch.quantile(y.reshape(y.shape[0], -1), q=0.99, dim=-1)[:,None,None])
y = y - y.reshape(y.shape[0], -1).min(-1).values[:,None,None]
y = y / y.reshape(y.shape[0], -1).max(-1).values[:,None,None]

res = y
if args.external > 0. or args.internal > 0.:
    res = gp(x, y, params)

###

pathlib.Path(f"resuls/{args.dataset}").mkdir(parents=True, exist_ok=True)
image.imsave(f"results/{args.dataset}/external({args.external})-internal({args.internal}).png", res[0], cmap='binary_r')
