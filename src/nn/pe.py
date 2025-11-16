import torch as t
import torch.nn as nn
import math

from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional
from pdb import set_trace 

def compute_trig(d_head, n_ctx, C):
    thetas = t.exp(-math.log(C)*t.arange(0,d_head,2)/d_head)
    thetas = thetas.repeat([2,1]).T.flatten()
    positions = t.arange(n_ctx)
    all_thetas = positions.unsqueeze(1)*thetas.unsqueeze(0)
    sins = t.sin(all_thetas)
    coss = t.cos(all_thetas)
    return sins, coss

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
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.C = C
        sins, coss = compute_trig(d_head, n_ctx, C)
        self.register_buffer('sins', sins.unsqueeze(0).unsqueeze(2))
        self.register_buffer('coss', coss.unsqueeze(0).unsqueeze(2))
    
    def forward(self, key_or_query: Float[Tensor, "batch sequence n_head d_head"],
                      offset: int = 0):
        x = key_or_query 
        x_perm = t.empty(x.shape, device=x.device, dtype=x.dtype) # batch sequence n_head d_head, we perm the last axis
        even = t.arange(0, x.shape[-1], 2)
        odd = t.arange(1, x.shape[-1],2)
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        assert x.shape[1] >= 1, f"x.shape[1] must be >= 1, got {x.shape}"
        return self.coss[:,offset:offset+x.shape[1]]*x + self.sins[:,offset:offset+x.shape[1]]*x_perm

class LearnRoPE(nn.Module):
    def __init__(self, d_head, n_ctx, C=10000):
        super().__init__()
        self.d_head = d_head
        self.n_ctx = n_ctx
        self.C = C
        sins, coss = compute_trig(d_head, n_ctx, C)
        self.sins = nn.Parameter(sins.unsqueeze(0).unsqueeze(2))
        self.coss = nn.Parameter(coss.unsqueeze(0).unsqueeze(2))
    
    def forward(self, key_or_query: Float[Tensor, "batch sequence n_head d_head"],
                      offset: int = 0):
        x = key_or_query 
        x_perm = t.empty(x.shape, device=x.device, dtype=x.dtype) # batch sequence n_head d_head, we perm the last axis
        even = t.arange(0, x.shape[-1], 2)
        odd = t.arange(1, x.shape[-1],2)
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        assert x.shape[1] >= 1, f"x.shape[1] must be >= 1, got {x.shape}"
        return self.coss[:,offset:offset+x.shape[1]]*x + self.sins[:,offset:offset+x.shape[1]]*x_perm

class VidRoPE(nn.Module):
    def __init__(self, d_head, 
                    d_x, 
                    d_y,
                    d_t,
                    ctx_x,
                    ctx_y,
                    ctx_t,
                    C_x,
                    C_y,
                    C_t,
                    toks_per_frame,
                    n_registers):
        super().__init__()
        assert d_x + d_y + d_t <= d_head, f"dx + dy + dt > d_head"
        self.d_head = d_head
        self.d_x = d_x
        self.d_y = d_y
        self.d_t = d_t
        self.ctx_x = ctx_x 
        self.ctx_y = ctx_y
        self.ctx_t = ctx_t 
        self.C_x = C_x
        self.C_y = C_y
        self.C_t = C_t
        self.toks_per_frame = toks_per_frame 
        self.n_registers = n_registers 
        sins_x, coss_x = compute_trig(d_x, ctx_x+1, C_x) # +1 for the register
        self.register_buffer("sins_x", sins_x.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_x", coss_x.unsqueeze(0).unsqueeze(2))
        sins_y, coss_y = compute_trig(d_y, ctx_y+1, C_y) # +1 for the register
        self.register_buffer("sins_y", sins_y.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_y", coss_y.unsqueeze(0).unsqueeze(2))
        sins_t, coss_t = compute_trig(d_t, ctx_t, C_t)
        self.register_buffer("sins_t", sins_t.unsqueeze(0).unsqueeze(2))
        self.register_buffer("coss_t", coss_t.unsqueeze(0).unsqueeze(2)) 
        n_frames = ctx_t
        # ctx_x should be equal to width
        # ctx_y should be equal to height
        pos_x = t.arange(self.ctx_x).repeat(self.ctx_y) # w cols with h entries each
        pos_x = t.cat([pos_x, t.tensor([self.ctx_x], dtype=t.int32)]) # deal with register
        pos_x = pos_x.repeat(n_frames)
        pos_y = t.arange(self.ctx_y).repeat_interleave(self.ctx_x) # h rows with w entries each
        pos_y = t.cat([pos_y, t.tensor([self.ctx_y], dtype=t.int32)]) # deal with register
        pos_y = pos_y.repeat(n_frames)
        pos_t = t.arange(n_frames).repeat_interleave(self.toks_per_frame)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.register_buffer("pos_t", pos_t)

    
    def rotate(self, x, pos_idcs, coss, sins):
        x_perm = t.empty(x.shape, device=x.device, dtype=x.dtype) # batch sequence n_head d_head, we perm the last axis
        even = t.arange(0, x.shape[-1], 2, device=x.device)
        odd = t.arange(1, x.shape[-1], 2, device=x.device)
        x_perm[:, :, :, even] = -x[:, :, :, odd]
        x_perm[:, :, :, odd] = x[:, :, :, even]
        assert x.shape[1] >= 1, f"x.shape[1] must be >= 1, got {x.shape}"
        assert pos_idcs.shape[0] == x.shape[1], f"pos_idcs length {pos_idcs.shape[0]} must match x.shape[1] {x.shape[1]}"

        return coss[:,pos_idcs]*x + sins[:,pos_idcs]*x_perm


    def forward(self, key_or_query: Float[Tensor, "batch sequence n_head d_head"],
                      offset: int = 0): 
        x = key_or_query 
        x[:, :, :, :self.d_x] = self.rotate(x[:, :, :, :self.d_x], self.pos_x, self.coss_x, self.sins_x) 
        x[:, :, :, self.d_x:self.d_x+self.d_y] = self.rotate(x[:, :, :, self.d_x:self.d_x+self.d_y], self.pos_y, self.coss_y, self.sins_y)
        x[:, :, :, self.d_x+self.d_y:self.d_x+self.d_y+self.d_t] = self.rotate(x[:, : , :, self.d_x+self.d_y:self.d_x+self.d_y+self.d_t], self.pos_t+(offset//self.toks_per_frame), self.coss_t, self.sins_t) 
        return x

        