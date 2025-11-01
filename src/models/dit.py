import torch as t
from torch import nn

from ..nn.attn import Attention
from ..nn.modulation import AdaLN, Gate
from ..nn.patch import Patch, UnPatch
from ..nn.geglu import GEGLU

from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional

class CausalBlock(nn.Module):
    def __init__(self, d_model, expansion, n_heads):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads

        self.norm1 = AdaLN(d_model)
        self.attn = Attention(d_model, n_heads)
        self.gate1 = Gate(d_model)
        self.norm2 = AdaLN(d_model)
        self.geglu = GEGLU(d_model, expansion*d_model, d_model)
        self.gate2 = Gate(d_model)
    
    def forward(self, zr, xa, cond, clean, mask_cross, mask_self):
        # batch durseq1 d
        # batch durseq2 d
        zr = self.norm1(zr, cond) 
        xa = self.norm1(xa, clean) 
        xkv = t.cat((zr, xa), dim=1)
        crossattn, _, _ = self.attn(zr, xkv, mask=mask_cross)
        selfattn, _, _ = self.attn(xa, xa, mask=mask_self)
        zr = self.gate1(zr, cond)
        xa = self.gate1(xa, clean)

        zr = self.norm2(zr + crossattn, cond)
        xa = self.norm2(xa + selfattn, clean)
        zr = self.geglu(zr)
        xa = self.geglu(xa)
        zr = self.gate2(zr, cond)
        xa = self.gate2(xa, clean)
        return zr, xa


class CausalDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, T, patch_size=2, n_heads=8, expansion=4, n_blocks=6, n_registers=1, n_actions=3):
        super().__init__()
        self.height = height
        self.width = width
        self.n_window = n_window
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.expansion = expansion
        self.n_registers = n_registers
        self.blocks = nn.ModuleList([CausalBlock(d_model, expansion, n_heads) for _ in range(n_blocks)])
        self.patch = Patch(out_channels=d_model)
        self.unpatch = UnPatch(height, width, in_channels=d_model)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.pe_grid = nn.Parameter(t.randn(height//patch_size*width//patch_size, d_model) * 1/d_model**0.5)
        self.pe_frames = nn.Parameter(t.randn(n_window, d_model) * 1/d_model**0.5)
        self.time_emb = nn.Embedding(T, d_model) # exchange to sinusoidal
    
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
        
        z = self.patch(z) # batch dur seq d
        z += self.pe_grid[None, None]
        #print(z.shape, self.pe_frames.shape)
        #z += self.pe_frames[None, :, None] # this does not work with absolute
        x = self.patch(frames) # batch dur seq d
        x += self.pe_grid[None, None]
        # x += self.pe_frames[None, :, None]
        a = self.action_emb(actions) # batch dur d
        # a += self.pe_frames[None]
        # self.registers is in 1x
        print('z', z.shape)
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        xa = t.cat((x, a[:, :-1].unsqueeze(2)), dim=2)
        mask_cross, mask_self = self.causal_mask(zr, xa)
        batch, durzr, seqzr, d = zr.shape
        print('zr', zr.shape)
        batch, durxa, seqxa, d = xa.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        xa = xa.reshape(batch, -1, d)
        cond = self.time_emb(ts)
        clean = self.time_emb(t.zeros((ts.shape[0], ts.shape[1]-1), dtype=ts.dtype, device=ts.device))
        for block in self.blocks:
            zr, xa = block(zr, xa, cond, clean, mask_cross, mask_self)
        print(zr.shape)
        zr = zr.reshape(batch, durzr, seqzr, d)
        print(zr.shape)
        out = self.unpatch(zr[:, :, :-self.n_registers])
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
        m_left = t.eye(zr.shape[1], dtype=bool)
        tmp = t.ones((zr.shape[1], zr.shape[1]), dtype=bool)
        m_right = t.tril(tmp, -1) ^ t.tril(tmp, -1 - self.n_window)
        m_right = m_right[:, :-1]

        # blow them up to dur_seq and durkv_seqkv size
        m_left = t.kron(m_left, t.ones((zr.shape[2], zr.shape[2]), dtype=bool))
        m_right = t.kron(m_right, t.ones((zr.shape[2], xa.shape[2]), dtype=bool))

        m_self = t.tril(t.ones((xa.shape[1], xa.shape[1]), dtype=bool))
        m_self = t.kron(m_self, t.ones((xa.shape[2],xa.shape[2]), dtype=bool))
        return ~t.cat((m_left, m_right), dim=1), ~m_self # we want to mask out the ones



if __name__ == "__main__":
    dit = CausalDit(10, 12, 3, 64, 5, n_blocks=2)
    frames = t.rand((2, 30*5-1, 3, 10, 12))
    z = t.rand((2, 30*5, 3, 10, 12))
    actions = t.randint(3, (2, 30*5))
    ts = t.randint(5, (2, 30*5))
    out = dit(z, frames, actions, ts)