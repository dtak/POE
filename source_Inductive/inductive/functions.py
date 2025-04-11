import torch

_NAMED_FUNCTIONS = {
    'cubed': lambda x: x**3,
    'quadratic_without_interaction': lambda x: x**2,
    'sine': lambda x: torch.sin(x),
    'quasi': lambda x: x + torch.sin(3 * x),
    'quadratic_quasi': lambda x: x**2 / 10 + torch.sin(3 * x),
    'exp_quasi': lambda x: torch.sin(torch.exp(x)),
    'exponential': lambda x: torch.exp(x)}

_NAMED_FUNCTIONS_DERIVATIVES = {
    'cubed': lambda x: 3 * (x**2),
    'quadratic_without_interaction': lambda x: 2 * x,
    'sine': lambda x: torch.cos(x),
    'quasi': lambda x: 1 + 3 * torch.cos(3 * x),
    'quadratic_quasi': lambda x: x / 5 + 3 * torch.cos(3 * x),
    'exp_quasi': lambda x: x + torch.exp(x) * torch.cos(torch.exp(x)),
    'exponential': lambda x: torch.exp(x)}

def create(named_function: str, dims: int = 3, seed: int = 42):
    if named_function not in _NAMED_FUNCTIONS:
        raise ValueError
    
    nabla_f = _NAMED_FUNCTIONS_DERIVATIVES[named_function]
    
    with torch.random.fork_rng():
        torch.manual_seed(seed)

        x = -5 + 10 * torch.rand(size=(1000, dims))
        y = torch.sum(nabla_f(x), dim=-1, keepdim=True)

        x_train = -5 + 10 * torch.rand(size=(1000, dims))
        y_train = torch.sum(nabla_f(x_train), dim=-1, keepdim=True)

    return x, y, x_train, y_train
