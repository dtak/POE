import torch
from typing import Literal

def gp(x: torch.Tensor, y: torch.Tensor, params: dict[str]) -> torch.Tensor:

    _K0 = torch.einsum('nij,nij->n', x, x)
    _K0 = _K0[:,None] + _K0[None,:] - 2 * torch.einsum('nij,Nij->nN', x, x)
    _K0 = _K0 / x.shape[1] / x.shape[2] / params['lengthscale_external']**2
    
    K0 = torch.exp(-0.5 * _K0)
    if "double" in params and params["double"]:
        K0 = K0.to(dtype=torch.double)

    _K1 = torch.arange(y.shape[1]) / y.shape[1] / params['lengthscale_internal']
    _K2 = torch.arange(y.shape[2]) / y.shape[2] / params['lengthscale_internal']

    if "double" in params and params["double"]:
        _K1 = _K1.to(dtype=torch.double)
        _K2 = _K2.to(dtype=torch.double)

    K1 = torch.exp(-0.5 * (_K1[:,None] - _K1[None,:])**2)
    K2 = torch.exp(-0.5 * (_K2[:,None] - _K2[None,:])**2)

    U0, S0, _ = torch.linalg.svd(K0)
    U1, S1, _ = torch.linalg.svd(K1)
    U2, S2, _ = torch.linalg.svd(K2)

    S0 = S0[:,None,None]
    S1 = S1[None,:,None]
    S2 = S2[None,None,:]
    
    S_kernel = 1 / (params['lambda_external'] / S0 + params['lambda_internal'] / S1 / S2)
    S_result = S_kernel / (params['noise'] + S_kernel)

    res = y
    if "double" in params and params["double"]:
        res = res.to(dtype=torch.double)

    res = torch.einsum('jJ,NIJ->NIj', U2.T, res)
    res = torch.einsum('iI,NIj->Nij', U1.T, res)
    res = torch.einsum('nN,Nij->nij', U0.T, res)
    res = res * S_result
    res = torch.einsum('jJ,NIJ->NIj', U2, res)
    res = torch.einsum('iI,NIj->Nij', U1, res)
    res = torch.einsum('nN,Nij->nij', U0, res)

    return res

def gp_robust(
        x: torch.Tensor,
        y: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        params: dict[str],
        mode: Literal['inductive', 'transductive']) -> torch.Tensor:
    
    if mode == 'transductive':
        x_train = torch.concatenate((x, x_train), dim=0)
        y_train = torch.concatenate((y, y_train), dim=0)

    _K = torch.einsum('n...,n...->n', x_train, x_train)
    _K = _K[:,None] + _K[None,:] - 2 * torch.einsum('n...,N...->nN', x_train, x_train)
    _K = _K / x_train.shape[1:].numel() / params['lengthscale']**2
    K = torch.exp(-0.5 * _K) / params['lambda']

    U, S, _ = torch.linalg.svd(K)
    KNOISE_INVERSE = torch.einsum('ns,s,sN->nN', U, 1 / (params['noise'] + S), U.T)

    if mode == 'transductive':
        L = K[:x.shape[0],:]

    elif mode == 'inductive':
        _L = torch.einsum('m...,m...->m', x, x)[:,None]
        _L = _L + torch.einsum('n...,n...->n', x_train, x_train)[None,:]
        _L = _L - 2 * torch.einsum('m...,n...->mn', x, x_train)
        _L = _L / x.shape[1:].numel() / params['lengthscale']**2
        L = torch.exp(-0.5 * _L) / params['lambda']

    res = torch.einsum('mn,nN,N...->m...', L, KNOISE_INVERSE, y_train)

    if mode == 'transductive':
        return res
    
    elif mode == 'inductive':
        res_variance = 1 / params['lambda'] - torch.einsum('mn,nN,mN->m', L, KNOISE_INVERSE, L)
        res_variance = res_variance.reshape(-1, *[1 for _ in y.shape[1:]])
        res = y / params['noise'] + res / res_variance
        res /= 1 / params['noise'] + 1 / res_variance
        return res
