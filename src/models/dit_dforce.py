import torch as t
from torch import nn
import torch.nn.functional as F

from ..nn.attn import Attention, AttentionSlow
from ..nn.modulation import AdaLN, Gate
from ..nn.patch import Patch, UnPatch, PatchCond, UnPatchCond
from ..nn.geglu import GEGLU
from ..nn.pe import FrameRoPE, NumericEncoding, RoPE
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional

import matplotlib.pyplot as plt
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = t.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = t.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = t.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = t.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = GEGLU(
            dim, ffn_dim_multiplier*dim, dim
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

class CausalBlock(nn.Module):
    def __init__(self, d_model, expansion, n_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads
        self.norm1 = AdaLN(d_model)
        if t.backends.mps.is_available():
            self.selfattn = AttentionSlow(d_model, n_heads, rope=rope)
        else:
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
    def __init__(self, height, width, n_window, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=4, bidirectional=False, debug=False, legacy=False):
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
        self.bidirectional = bidirectional
        self.toks_per_frame = height//patch_size*width//patch_size + n_registers
        #self.rope_seq = RoPE(d_model//n_heads, self.n_window*self.toks_per_frame)
        #self.blocks = nn.ModuleList([CausalBlock(d_model, expansion, n_heads, rope=self.rope_seq) for _ in range(n_blocks)])
        self.patch = Patch(in_channels=in_channels, out_channels=d_model, patch_size=patch_size)
        self.norm = AdaLN(d_model)
        self.unpatch = UnPatch(height, width, in_channels=d_model, out_channels=in_channels, patch_size=patch_size)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.time_emb = NumericEncoding(dim=d_model, n_max=T)
        self.time_emb_mixer = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(layer_id, d_model, n_heads, 1., expansion, 1e-5) for layer_id in range(n_blocks)])
        self.register_buffer("freqs_cis", CausalDit.precompute_freqs_cis(d_model // n_heads, 4096))
    
    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (t.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t_ = t.arange(end)
        freqs = t.outer(t_, freqs).float()
        freqs_cis = t.polar(t.ones_like(freqs), freqs)
        return freqs_cis
    
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
        cond = cond[:,0,:]
        z = self.patch(z) # batch dur seq d

        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        if self.bidirectional:
            mask_self = None
        else:
            mask_self = self.causal_mask(zr)
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d
        
        for block in self.blocks:
            #zr = block(zr, cond, mask_self)
            zr = block(zr, self.freqs_cis, cond)
        zr = self.norm(zr, cond.unsqueeze(1))
        zr = zr.reshape(batch, durzr, seqzr, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
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

def get_model(height, width, n_window=5, d_model=64, T=100, n_blocks=2, patch_size=2, n_heads=8, bidirectional=False, in_channels=3):
    return CausalDit(height, width, n_window, d_model, T, in_channels=in_channels, n_blocks=n_blocks, patch_size=patch_size, n_heads=n_heads, bidirectional=bidirectional)

if __name__ == "__main__":
    dit = CausalDit(20, 20, 100, 64, 5, n_blocks=2)
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(4, (2, 6))
    ts = t.rand((2, 6))
    out = dit(z, actions, ts)
    print(z.shape)
    print(out.shape)