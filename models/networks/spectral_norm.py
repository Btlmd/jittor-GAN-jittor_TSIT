"""
Note: This file is copied from https://github.com/Lewis-Liang/PytorchAndJittor
Reference: https://github.com/Lewis-Liang/PytorchAndJittor/blob/a3c7bcfe898a936857cdfc103c205265695453d0/jt_spectral_norm.py
"""

import jittor as jt
from jittor.misc import normalize
from typing import Any, Optional, TypeVar
import jittor.nn as nn
from jittor.nn import Module

class SpectralNorm:
    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: jt.Var) -> jt.Var:
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> jt.Var:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with jt.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(nn.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = normalize(nn.matmul(weight_mat, v), dim=0, eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = jt.matmul(u, jt.matmul(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        with jt.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        setattr(module, self.name, jt.Var(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.is_training()))


    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = jt.matmul(jt.matmul(weight_mat.t().mm(weight_mat).pinverse(), jt.matmul(weight_mat.t(), u.unsqueeze(1)))).squeeze(1)
        return v.mul_(target_sigma / jt.matmul(u, jt.matmul(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        with jt.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # initialize `u` and `v`
            import numpy.random as random
            u_np = random.normal(0, 1, size=[h])
            v_np = random.normal(0, 1, size=[w])
            u = normalize(jt.Var(u_np), dim=0, eps=fn.eps)
            v = normalize(jt.Var(v_np), dim=0, eps=fn.eps)

        delattr(module, fn.name)

        import numpy.random as random
        weight_ = jt.randn_like(weight)
        setattr(module, fn.name + "_orig", jt.Var(weight_))
        setattr(module, fn.name, jt.Var(weight_))
        setattr(module, fn.name + "_u", u)
        setattr(module, fn.name + "_v", v)
        module.register_pre_forward_hook(fn)
        return fn


T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    if dim is None:
        if isinstance(module, (nn.ConvTranspose,
                               nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

