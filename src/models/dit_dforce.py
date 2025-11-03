import torch as t
from torch import nn

from ..nn.attn import Attention
from ..nn.modulation import AdaLN, Gate
from ..nn.patch import Patch, UnPatch, PatchCond, UnPatchCond
from ..nn.geglu import GEGLU
from ..nn.pe import FrameRoPE, NumericEncoding
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional

import matplotlib.pyplot as plt
import math

class CausalBlock(nn.Module):
    def __init__(self, d_model, expansion, n_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads
        self.norm1 = AdaLN(d_model)
        self.selfattn = Attention(d_model, n_heads, rope=rope)
        self.gate1 = Gate(d_model)
        self.norm2 = AdaLN(d_model)
        self.geglu = GEGLU(d_model, expansion*d_model, d_model)
        self.gate2 = Gate(d_model)
    
    def forward(self, z, cond, mask_self):
        # batch durseq1 d
        # batch durseq2 d
        residual = z
        z = self.norm1(z, cond)
        z, _, _ = self.selfattn(z, z, mask=mask_self)
        z = self.gate1(z, cond)
        z = z + residual

        residual = z
        z = self.norm2(z, cond)
        z = self.geglu(z)
        z = self.gate2(z, cond)
        z = z + residual
        return z


class CausalDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, T=1000, 
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=4, nctx=20000, debug=False, legacy=False):
        super().__init__()
        self.height = height
        self.width = width
        self.n_window = n_window
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.expansion = expansion
        self.n_registers = n_registers
        self.T = T
        self.patch_size = patch_size
        self.debug = debug
        self.legacy = legacy
        self.frame_rope = FrameRoPE(d_model//n_heads, nctx, height//patch_size*width//patch_size + n_registers)
        self.blocks = nn.ModuleList([CausalBlock(d_model, expansion, n_heads, rope=self.frame_rope) for _ in range(n_blocks)])
        self.patch = Patch(out_channels=d_model, patch_size=patch_size)
        self.unpatch = UnPatchCond(height, width, in_channels=d_model, patch_size=patch_size)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.pe_grid = nn.Parameter(t.randn(height//patch_size*width//patch_size, d_model) * 1/d_model**0.5)
        self.time_emb = NumericEncoding(dim=d_model, n_max=T)
    
    def forward(self, 
                z: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"]):
 
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])

        a = self.action_emb(actions) # batch dur d
        ts_scaled = (ts * self.T).clamp(0, self.T - 1).long()
        cond = self.time_emb(ts_scaled) 
        cond += a

        z = self.patch(z) # batch dur seq d
        z += self.pe_grid[None, None]

        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        mask_self = self.causal_mask(zr)
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        
        for block in self.blocks:
            zr = block(zr, cond, mask_self)
        zr = zr.reshape(batch, durzr, seqzr, d)

        out = self.unpatch(zr[:, :, :-self.n_registers], cond)
        return out # batch dur channels height width
    
    def causal_mask(self, z):
        m_self = t.tril(t.ones((z.shape[1], z.shape[1]), dtype=t.int8, device=self.device))
        m_self = t.kron(m_self, t.ones((z.shape[2], z.shape[2]), dtype=t.int8, device=self.device))
        m_self = m_self.to(bool)
        return ~m_self # we want to mask out the ones
    
    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype

def get_model(height, width, n_window=5, d_model=64, T=100, n_blocks=2, patch_size=2, n_heads=8):
    return CausalDit(height, width, n_window, d_model, T, n_blocks=n_blocks, patch_size=patch_size, n_heads=n_heads)

if __name__ == "__main__":
    dit = CausalDit(20, 20, 3, 64, 5, n_blocks=2)
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(4, (2, 6))
    ts = t.rand((2, 6))
    out = dit(z, actions, ts)
    print(z.shape)
    print(out.shape)