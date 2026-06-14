"""
Adapted from 
https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
https://github.com/seongq/flowmse/blob/main/flowmse/odes.py 
"""
from scipy import integrate
import torch

from .odesolvers import ODEsolver, ODEsolverRegistry

import numpy as np

__all__ = [
    'ODEsolverRegistry', 'ODEsolver', 'get_sampler'
]

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def get_white_box_solver(
    odesolver_name, ode, VF_fn, y, context, t_eps=0.03, **kwargs
):
   
    odesolver_cls = ODEsolverRegistry.get_by_name(odesolver_name)
    odesolver = odesolver_cls(ode, VF_fn)

    def ode_solver():
        """The PC sampler function."""
        with torch.no_grad():
            xt = ode.prior_sampling(y.shape, y).to(y.device)
            if odesolver_name=="euler":
                timesteps = torch.linspace(ode.T, t_eps, ode.N, device=y.device)
            xt = xt.to(y.device)
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt = odesolver.update_fn(xt, vec_t, y, context, stepsize)
            x_result = xt
            ns = len(timesteps)
            return x_result, ns
    
    return ode_solver
