from torch import nn
import torch as t
from torch import Tensor
from jaxtyping import Float

class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(t.ones(d))
        self.b = nn.Parameter(t.zeros(d))

    def forward(self, residual: Float[Tensor, "batch dur seq d_model"]) -> Float[Tensor, "batch dur seq d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b