import torch as t
from torch import nn
import torch.nn.functional as F

from ..nn.attn import AttentionEinOps, KVCache, KVCacheNaive
from ..nn.attn2 import Attention, create_block_mask, create_block_causal_mask_mod
from ..nn.patch import Patch, UnPatch
from ..nn.geglu import GEGLU
from ..nn.pe import NumericEncoding, RoPE, LearnRoPE, VidRoPE
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional, Literal

import matplotlib.pyplot as plt
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CausalBlock(nn.Module):
    def __init__(self, layer_idx, d_model, expansion, n_heads, rope=None, ln_first = False, use_flex=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.selfattn = Attention(d_model, n_heads, rope=rope, use_flex=use_flex)
        self.norm2 = nn.LayerNorm(d_model)
        self.geglu = GEGLU(d_model, expansion*d_model, d_model)
        
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
    
    def forward(self, z, cond, mask_self, cached_k=None, cached_v=None):
        # batch durseq1 d
        # batch durseq2 d
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = z
        z = modulate(self.norm1(z), mu1, sigma1)
        z = z.to(dtype=self.dtype)
        z, k_new, v_new = self.selfattn(z, z, mask=mask_self, k_cache=cached_k, v_cache=cached_v)            
        z = residual + c1*z

        residual = z
        z = modulate(self.norm2(z), mu2, sigma2)
        z = z.to(dtype=self.dtype)
        z = self.geglu(z)
        z = residual + c2*z
        return z, k_new, v_new
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype
    
    @property
    def device(self):
        return self.parameters().__next__().device

class CausalDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=4, bidirectional=False, 
                       debug=False, 
                       rope_C=10000,
                       rope_tmax=None,
                       rope_type: Literal["rope", "learn", "vid"] = "rope",
                       ln_first: bool = False,
                       use_flex: bool = False):
        super().__init__()
        self.height = height
        self.width = width
        self.n_window = n_window
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.n_blocks = n_blocks
        self.expansion = expansion
        self.n_registers = n_registers
        self.T = T
        self.patch_size = patch_size
        self.debug = debug
        self.bidirectional = bidirectional
        self.toks_per_frame = (height//patch_size)*(width//patch_size) + n_registers
        self.rope_C = rope_C
        self.use_flex = use_flex
        if rope_tmax is None:
            rope_tmax = self.n_window*self.toks_per_frame
        if rope_type == "rope":
            self.rope_seq = RoPE(d_model//n_heads, rope_tmax, C=rope_C)
        elif rope_type == "learn":
            self.rope_seq = LearnRoPE(d_model//n_heads, rope_tmax, C=rope_C)
        elif rope_type == "vid":
            d_head = d_model//n_heads 
            d_x = d_y = d_t = d_head // 3
            C_x = C_y = C_t = rope_C // 3
            ctx_x = width // patch_size
            ctx_y = height // patch_size
            ctx_t = self.n_window
            self.rope_seq = VidRoPE(d_head, 
                                    d_x, 
                                    d_y,
                                    d_t,
                                    ctx_x,
                                    ctx_y,
                                    ctx_t,
                                    C_x,
                                    C_y,
                                    C_t,
                                    self.toks_per_frame,
                                    n_registers)

        self.grid_pe = None
        self.rope_tmax = rope_tmax

        self.blocks = nn.ModuleList([CausalBlock(lidx, d_model, expansion, n_heads, rope=self.rope_seq, ln_first=ln_first, use_flex=use_flex) for lidx in range(n_blocks)])
        self.patch = Patch(in_channels=in_channels, out_channels=d_model, patch_size=patch_size)
        self.norm = nn.LayerNorm(d_model)
        self.unpatch = UnPatch(height, width, in_channels=d_model, out_channels=in_channels, patch_size=patch_size)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.time_emb = NumericEncoding(dim=d_model, n_max=T)
        self.time_emb_mixer = nn.Linear(d_model, d_model)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        self.cache = None
        if not self.bidirectional:
            self.register_buffer("mask", self.causal_mask())
        else:
            self.register_buffer("mask", None)
    
    def create_cache(self, batch_size):
        return KVCache(batch_size, self.n_blocks, self.n_heads, self.d_head, self.toks_per_frame, self.n_window, dtype=self.dtype, device=self.device)

    def create_cache2(self, batch_size):
        return KVCacheNaive(batch_size, self.n_blocks, self.n_heads, self.d_head, self.toks_per_frame, self.n_window, dtype=self.dtype, device=self.device)
    
    def forward(self, 
                z: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"],
                cached_k: Optional[Float[Tensor, "layer batch dur seq d"]] = None,
                cached_v: Optional[Float[Tensor, "layer batch dur seq d"]] = None):
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])

        a = self.action_emb(actions) # batch dur d
        ts_scaled = (ts * self.T).clamp(0, self.T - 1).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        cond = cond.repeat_interleave(self.toks_per_frame, dim=1)
        z = self.patch(z) # batch dur seq d
        if self.grid_pe is not None:
            z = z + self.grid_pe[None, None]

        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        if self.bidirectional:
            mask_self = None
        else:
            mask_self = self.mask
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        
        k_update = []
        v_update = []
        for bidx, block in enumerate(self.blocks):
            ks = cached_k[bidx] if cached_k is not None else None 
            vs = cached_v[bidx] if cached_v is not None else None
            zr, k_new, v_new = block(zr, cond, mask_self, cached_k=ks, cached_v=vs)
            if k_new is not None:
                k_update.append(k_new.unsqueeze(0))
                v_update.append(v_new.unsqueeze(0))
        if len(k_update) > 0:
            k_update = t.cat(k_update, dim=0)
            v_update = t.cat(v_update, dim=0)

        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, durzr, seqzr, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
        return out, k_update, v_update
    
    def causal_mask(self):
        if self.use_flex:
            return create_block_mask(create_block_causal_mask_mod(self.toks_per_frame), B=None, H=None, Q_LEN=self.toks_per_frame*self.n_window, KV_LEN=self.toks_per_frame*self.n_window)
        else:
            size = self.n_window
            m_self = t.tril(t.ones((size, size), dtype=t.int8, device=self.device)) # - t.tril(t.ones((size, size), dtype=t.int8, device=self.device), diagonal=-self.n_window) # this would be useful if we go bigger than windowxwindow
            m_self = t.kron(m_self, t.ones((self.toks_per_frame, self.toks_per_frame), dtype=t.int8, device=self.device))
            m_self = m_self.to(bool)
            return m_self
    
    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype


def get_model(height, width, 
              n_window=5, 
              d_model=64, 
              T=100, 
              n_blocks=2, 
              patch_size=2, 
              n_heads=8, 
              bidirectional=False, 
              in_channels=3, 
              C=10000, 
              rope_type: Literal["rope", "learn", "vid"] = "rope",
              ln_first=False,
              use_flex=False):
    return CausalDit(height, width, 
                     n_window, 
                     d_model, 
                     T, 
                     in_channels=in_channels, 
                     n_blocks=n_blocks, 
                     patch_size=patch_size, 
                     n_heads=n_heads, 
                     bidirectional=bidirectional, 
                     rope_C=C, 
                     rope_type=rope_type,
                     ln_first=ln_first,
                     use_flex=use_flex)

if __name__ == "__main__":
    print("running w/o cache")
    dit = CausalDit(20, 20, 100, 64, 5, n_blocks=2)
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(4, (2, 6))
    ts = t.rand((2, 6))
    out, _, _ = dit(z, actions, ts)
    print(z.shape)
    print(out.shape)