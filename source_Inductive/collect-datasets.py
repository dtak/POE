import torch
from inductive.datasets import load_imagenet_resnet50
from inductive.datasets import create_perturbations

raw_x, raw_y = load_imagenet_resnet50(sample=2000)
torch.save((raw_x, raw_y), f"datasets/imagenet2000.pt")

for i in range(5):
    raw_x, raw_y = load_imagenet_resnet50(sample=2000, sample_seed=i)
    torch.save((raw_x, raw_y), f"datasets/imagenet2000-seed{i}.pt")

raw_x, raw_y = create_perturbations(i=0)
torch.save((raw_x, raw_y), f"datasets/perturbations-sample0.pt")

raw_x, raw_y = create_perturbations(i=2)
torch.save((raw_x, raw_y), f"datasets/perturbations-sample2.pt")

for delta in [0.01, 0.05, 0.25, 1.25]:

    raw_x, raw_y = create_perturbations(i=0, delta=delta)
    torch.save((raw_x, raw_y), f"datasets/perturbations-sample0-delta({delta}).pt")

    raw_x, raw_y = create_perturbations(i=2, delta=delta)
    torch.save((raw_x, raw_y), f"datasets/perturbations-sample2-delta({delta}).pt")