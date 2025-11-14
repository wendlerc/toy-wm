import torch as t
import torch.nn as nn
import math

from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional

class NumericEncoding(nn.Module):
    def __init__(self, C = 1e4, dim = 64, n_max = 10000):
        super().__init__()
        args = t.exp(- math.log(C) * t.arange(0, dim, 2)/dim)
        args = t.arange(n_max)[:, None] * args[None, :]
        sins = t.sin(args)
        coss = t.cos(args)
        pe = t.empty((n_max, dim))
        pe[:,::2] = sins
        pe[:,1::2] = coss
        self.register_buffer("pe", pe)
    
    def forward(self, num):
        """
        expects integers between 0 and n_max
        """
        assert num.dtype == t.int32 or num.dtype == t.int64, f"wrong dtype {num.dtype}"
        return self.pe[num]


class RoPE(nn.Module):
    def __init__(self, d_head, n_ctx, C=10000):
        super().__init__()
        thetas = t.exp(-math.log(C)*t.arange(0,d_head,2)/d_head)
        thetas = thetas.repeat([2,1]).T.flatten()
        positions = t.arange(n_ctx)
        all_thetas = positions.unsqueeze(1)*thetas.unsqueeze(0)
        sins = t.sin(all_thetas)
        coss = t.cos(all_thetas)
        self.register_buffer('sins', sins.unsqueeze(0).unsqueeze(2))
        self.register_buffer('coss', coss.unsqueeze(0).unsqueeze(2))
    
    def forward(self, key_or_query: Float[Tensor, "batch sequence n_head d_head"],
                      offset: int = 0):
        x = key_or_query
        # start with doing it for just a single position m  
        x_perm = t.empty(x.shape, device=x.device, dtype=x.dtype) # batch sequence n_head d_head, we perm the last axis
        even = t.arange(0, x.shape[-1], 2)
        odd = t.arange(1, x.shape[-1],2)
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        assert x.shape[1] >= 1, f"x.shape[1] must be >= 1, got {x.shape}"
        return self.coss[:,offset:offset+x.shape[1]]*x + self.sins[:,offset:offset+x.shape[1]]*x_perm

