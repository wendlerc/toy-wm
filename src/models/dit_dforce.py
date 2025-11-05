import torch as t
from torch import nn
import torch.nn.functional as F

from ..nn.attn import Attention, AttentionEinOps, KVCache
from ..nn.patch import Patch, UnPatch
from ..nn.geglu import GEGLU
from ..nn.pe import FrameRoPE, NumericEncoding, RoPE
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional

import matplotlib.pyplot as plt
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CausalBlock(nn.Module):
    def __init__(self, layer_idx, d_model, expansion, n_heads, rope=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        if t.backends.mps.is_available():
            self.selfattn = AttentionEinOps(d_model, n_heads, rope=rope)
        else:
            self.selfattn = AttentionEinOps(d_model, n_heads, rope=rope) # there is a problem with flexattn i think
        self.norm2 = nn.LayerNorm(d_model)
        self.geglu = GEGLU(d_model, expansion*d_model, d_model)
        
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
    
    def forward(self, z, cond, mask_self, cache: Optional[KVCache] = None):
        # batch durseq1 d
        # batch durseq2 d
        mu1, sigma1, c1, mu2, sigma2, c2 = self.modulation(cond).chunk(6, dim=-1)
        residual = z
        z = modulate(self.norm1(z), mu1, sigma1)
        if cache is not None:
            k, v = cache.get(self.layer_idx)
            global_offset = cache.global_location
            z, k_new, v_new = self.selfattn(z, z, mask=mask_self, k_cache=k, v_cache=v, offset=global_offset)
            cache.extend(self.layer_idx, k_new, v_new)
        z = residual + c1*z

        residual = z
        z = modulate(self.norm2(z), mu2, sigma2)
        z = self.geglu(z)
        z = residual + c2*z
        return z


class CausalDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=4, bidirectional=False, 
                       debug=False, 
                       legacy=False,
                       frame_rope=False,
                       rope_C=10000,
                       rope_tmax=None):
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
        self.legacy = legacy
        self.bidirectional = bidirectional
        self.frame_rope = frame_rope
        self.toks_per_frame = (height//patch_size)*(width//patch_size) + n_registers
        if frame_rope:
            print("Using frame rope")
            print(self.toks_per_frame)
            self.rope_seq = FrameRoPE(d_model//n_heads, self.n_window, self.toks_per_frame, C=C)
            self.grid_pe = nn.Parameter(t.randn(self.toks_per_frame - n_registers, d_model) * 1/d_model**0.5)
        else:
            if rope_tmax is None:
                rope_tmax = 30*self.n_window*self.toks_per_frame
            self.rope_seq = RoPE(d_model//n_heads, rope_tmax, C=rope_C)
            self.grid_pe = None
        self.blocks = nn.ModuleList([CausalBlock(lidx, d_model, expansion, n_heads, rope=self.rope_seq) for lidx in range(n_blocks)])
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
    
    def activate_caching(self, batch_size):
        self.cache = KVCache(batch_size, self.n_blocks, self.n_heads, self.d_head, self.toks_per_frame, self.n_window)

    def deactivate_caching(self):
        self.cache = None
    
    def forward(self, 
                z: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"]):
 
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])

        a = self.action_emb(actions) # batch dur d
        ts_scaled = (ts * self.T).clamp(0, self.T - 1).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        #print(ts_scaled.shape, a.shape, cond.shape, actions.shape)
        cond = cond.repeat_interleave(self.toks_per_frame, dim=1)
        z = self.patch(z) # batch dur seq d
        if self.grid_pe is not None:
            z = z + self.grid_pe[None, None]
        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        if self.bidirectional:
            mask_self = None
        else:
            mask_self = self.causal_mask
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        
        for block in self.blocks:
            zr = block(zr, cond, mask_self, cache=self.cache)
        mu, sigma = self.modulation(cond).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, durzr, seqzr, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
        return out # batch dur channels height width
    
    @property
    def causal_mask(self):
        m_self = t.tril(t.ones((self.n_window+1, self.n_window+1), dtype=t.int8, device=self.device))
        m_self = t.kron(m_self, t.ones((self.toks_per_frame, self.toks_per_frame), dtype=t.int8, device=self.device))
        m_self = m_self.to(bool)
        return ~ m_self # we want to mask out the ones
    
    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype

def get_model(height, width, n_window=5, d_model=64, T=100, n_blocks=2, patch_size=2, n_heads=8, bidirectional=False, in_channels=3, frame_rope=False, C=10000):
    return CausalDit(height, width, n_window, d_model, T, in_channels=in_channels, n_blocks=n_blocks, patch_size=patch_size, n_heads=n_heads, bidirectional=bidirectional, frame_rope=frame_rope, C=C)

if __name__ == "__main__":
    print("running w/o cache")
    dit = CausalDit(20, 20, 100, 64, 5, n_blocks=2)
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(4, (2, 6))
    ts = t.rand((2, 6))
    out = dit(z, actions, ts)
    print(z.shape)
    print(out.shape)

    print("running w cache")
    dit = CausalDit(20, 20, 10, 64, 5, n_blocks=2)
    dit.activate_caching(2)
    print(dit.cache.toks_per_frame)
    print(dit.cache.size)
    for i in range(30):
        print(dit.cache.local_loc)
        print(dit.cache.global_loc)
        z = t.rand((2, 1, 3, 20, 20))
        actions = t.randint(4, (2, 1))
        ts = t.rand((2, 1))
        out = dit(z, actions, ts)
        print(i, z.shape)
        print(i, out.shape)