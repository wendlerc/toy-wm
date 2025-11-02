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
        self.crossattn = Attention(d_model, n_heads, rope=rope)
        self.selfattn = Attention(d_model, n_heads, rope=rope)
        self.gate1 = Gate(d_model)
        self.norm2 = AdaLN(d_model)
        self.geglu = GEGLU(d_model, expansion*d_model, d_model)
        self.gate2 = Gate(d_model)
    
    def forward(self, zr, xa, cond, clean, mask_cross, mask_self):
        # batch durseq1 d
        # batch durseq2 d
        residual_zr = zr
        residual_xa = xa
        zr = self.norm1(zr, cond)
        xa = self.norm1(xa, clean)
        xkv = t.cat((zr , xa), dim=1)
        zr, _, _ = self.crossattn(zr, xkv, mask=mask_cross)
        xa, _, _ = self.selfattn(xa, xa, mask=mask_self)
        zr = self.gate1(zr, cond)
        xa = self.gate1(xa, clean)
        zr = zr + residual_zr
        xa = xa + residual_xa

        residual_zr = zr
        residual_xa = xa
        zr = self.norm2(zr, cond)
        xa = self.norm2(xa, clean)
        zr = self.geglu(zr)
        xa = self.geglu(xa)
        zr = self.gate2(zr, cond)
        xa = self.gate2(xa, clean)
        zr = zr + residual_zr
        xa = xa + residual_xa
        return zr, xa


class CausalDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, T, 
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=3, nctx=20000, debug=False, legacy=False):
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
        if self.legacy:
            self.patch = Patch(out_channels=d_model, patch_size=patch_size)
            self.unpatch = UnPatch(height, width, in_channels=d_model, patch_size=patch_size)
        else:
            self.patch = PatchCond(out_channels=d_model, patch_size=patch_size)
            self.unpatch = UnPatchCond(height, width, in_channels=d_model, patch_size=patch_size)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.pe_grid = nn.Parameter(t.randn(height//patch_size*width//patch_size, d_model) * 1/d_model**0.5)
        if self.legacy:
            self.time_emb = nn.Embedding(T, d_model) # exchange to sinusoidal
        else:
            self.time_emb = NumericEncoding(dim=d_model, n_max=T)
    
    def forward(self, 
                z: Float[Tensor, "batch dur channels height width"], 
                frames: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"]):
                # dur is fps * s frames
            
        # we basically want for a window size of 3
        # z1 | f1 z2 | f1 f2 z3 as inputs and 
        # f1 | f2    | f3 as outputs
        # also we want all of this in the same forward pass if possible

        # it can be achieved with clever masking
        # z1, z2, z3 | f1, f2 (input)
        # f1, f2, f3 (output)

        #     z1, z2, z3, f1, f2
        # z1   1,  0,  0,  0,  0
        # z2   0,  1,  0,  1,  0
        # z3   0,  0,  1,  1,  1
        # crossatention (above) and a small self attn (below)
        # f1   0,  0,  0,  1,  0
        # f2   0,  0,  0,  1,  1
        # using self attention with queries (z1,z2,z3) and keys,vals (z1,z2,z3,f1,f2)
        # mask
        # (Id3 | downshifted by 1 autoregressive mask)
        # incorporating actions by adding them to the frames
        # z1, z2, z3, f1a1, f2a2
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])

        a = self.action_emb(actions) # batch dur d
        cond = self.time_emb((ts * self.T).long()) 
        if self.legacy:
            cond += a
        else:
            cond[:, 1:] += a[:, :-1]
        clean = self.time_emb(t.zeros((ts.shape[0], ts.shape[1]-1), dtype=t.long, device=ts.device)) + a[:, :-1]

        if self.legacy:
            z = self.patch(z) # batch dur seq d
        else:
            z = self.patch(z, cond)
        z += self.pe_grid[None, None]
        if self.legacy:
            x = self.patch(frames) # batch dur seq d
        else:
            x = self.patch(frames, clean)
        x += self.pe_grid[None, None]

        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        xa = t.cat((x, self.registers[None, None].repeat([x.shape[0], x.shape[1], 1, 1])), dim=2)
        mask_cross, mask_self = self.causal_mask(zr, xa)
        batch, durzr, seqzr, d = zr.shape
        batch, durxa, seqxa, d = xa.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        xa = xa.reshape(batch, -1, d)
        
        for block in self.blocks:
            zr, xa = block(zr, xa, cond, clean, mask_cross, mask_self)
        zr = zr.reshape(batch, durzr, seqzr, d)
        if self.legacy:
            out = self.unpatch(zr[:, :, :-self.n_registers])
        else:
            out = self.unpatch(zr[:, :, :-self.n_registers], cond)
        return out # batch dur channels height width
    
    def causal_mask(self, zr, xa):
        #     z1, z2, z3, z4, z5, f1, f2, f3, f4
        # z1   1,  0,  0,  0,  0,  0,  0,  0,  0
        # z2   0,  1,  0,  0,  0,  1,  0,  0,  0
        # z3   0,  0,  1,  0,  0,  1,  1,  0,  0
        # z4   0,  0,  0,  1,  0,  1,  1,  1,  0
        # z5   0,  0,  0,  0,  1,  0,  1,  1,  1
        # using cross attention with queries (z1,z2,z3) and keys,vals (z1,z2,z3,f1,f2)
        # mask
        # (Id3 | downshifted by 1 autoregressive mask)
        # incorporating actions by adding them to the frames
        # z1, z2, z3, f1a1, f2a2

        # zr: Float[Tensor, "batch dur seqzr d"], 
        # xa: Float[Tensor, "batch dur seqxa d"], 
        m_left = t.eye(zr.shape[1], dtype=t.int8, device=self.device)
        tmp = t.ones((zr.shape[1], zr.shape[1]), dtype=t.int8, device=self.device)
        m_right = t.tril(tmp, -1) - t.tril(tmp, -1 - self.n_window)
        m_right = m_right[:, :-1]

        # blow them up to dur_seq and durkv_seqkv size
        m_left = t.kron(m_left, t.ones((zr.shape[2], zr.shape[2]), dtype=t.int8, device=self.device))
        m_right = t.kron(m_right, t.ones((zr.shape[2], xa.shape[2]), dtype=t.int8, device=self.device))
        m_self = t.tril(t.ones((xa.shape[1], xa.shape[1]), dtype=t.int8, device=self.device))
        m_self = t.kron(m_self, t.ones((xa.shape[2],xa.shape[2]), dtype=t.int8, device=self.device))
        m_cross = t.cat((m_left, m_right), dim=1)
        m_cross = m_cross.to(bool)
        m_self = m_self.to(bool)
        if self.debug:
            plt.imshow(m_cross.numpy())
            plt.show()
            plt.imshow(m_self.numpy())
            plt.show()
        return ~m_cross, ~m_self # we want to mask out the ones
    
    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype

def get_model(height, width, n_window=5, d_model=64, T=100, n_blocks=2, patch_size=2):
    return CausalDit(height, width, n_window, d_model, T, n_blocks=n_blocks, patch_size=patch_size)

if __name__ == "__main__":
    dit = CausalDit(20, 20, 3, 64, 5, n_blocks=2)
    frames = t.rand((2, 6-1, 3, 20, 20))
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(3, (2, 6))
    ts = t.rand((2, 6))
    out = dit(z, frames, actions, ts)