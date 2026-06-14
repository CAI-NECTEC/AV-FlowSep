"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
ODE: https://github.com/seongq/flowmse
"""
import abc
import warnings
import math
import scipy.special as sc
import numpy as np
import torch
from src.utils.registry import Registry

ODERegistry = Registry("ODE")

class ODE(abc.ABC):
    """ODE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):        
        super().__init__()
        
    @abc.abstractmethod
    def ode(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    # @staticmethod
    # @abc.abstractmethod
    # def add_argparse_args(parent_parser):
    #     """
    #     Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
    #     """
    #     pass

    @abc.abstractmethod
    def copy(self):
        pass

@ODERegistry.register("flowmatching")
class FLOWMATCHING(ODE):
    def __init__(self, sigma_min=0.00, sigma_max=0.487, N=30, **ignored_kwargs):
        super().__init__()        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
    
    def copy(self):
        return FLOWMATCHING(self.sigma_min, self.sigma_max, self.N)
    
    @property
    def T(self):
        return 1
        
    def ode(self, x, t, *args):
        pass    
        
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):
        return (1-t)*self.sigma_min + t*self.sigma_max

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        xt = y + z * std[:, None, None, None]
        return xt

    def der_mean(self, x0, t, y):
        return y - x0
        
    def der_std(self, t):
        
        return self.sigma_max - self.sigma_min

@ODERegistry.register("without_latent")
class WithoutLatent(ODE):
    def __init__(self, N=30, **ignored_kwargs):
        super().__init__()        
        self.N = N
    
    def copy(self):
        return WithoutLatent(self.N)
    
    @property
    def T(self):
        return 1
        
    def ode(self,x,t,*args):
        pass    
        
    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):
        return t * 0

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        xt = y 
        return xt

    def der_mean(self,x0,t,y):
        return y-x0
        
    def der_std(self,t):
        return 0
        
@ODERegistry.register("with_latent")
class WithLatent(ODE):
    def __init__(self, alpha=0.25, N=30, **ignored_kwargs):
        super().__init__()   
        self.alpha = alpha
        self.N = N
        
    def copy(self):
        return WithLatent(self.alpha, self.N)
    
    @property
    def T(self):
        return 1
        
    def ode(self, x, t, *args):
        pass      

    def _mean(self, x0, t, y):       
        return (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*y

    def _std(self, t):
        return self.alpha * torch.sqrt(2*t * (1-t))

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t) 

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        xt = y
        return xt
        
    def der_mean(self, x0, t, y):
        return y - x0

    def der_std(self, t):
        t = t[:, None, None, None]
        return self.alpha * (1 - (2*t)) / torch.sqrt(2*t * (1-t))
