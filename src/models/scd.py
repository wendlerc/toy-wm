from numpy.ma.extras import mask_cols
import torch as t
from torch import nn
import torch.nn.functional as F

from ..nn.attn import Attention, create_block_mask, create_block_causal_mask_mod, KVCache, KVCacheNaive
from ..nn.patch import Patch, UnPatch
from ..nn.geglu import GEGLU
from ..nn.pe import NumericEncoding, RoPE, LearnRoPE, VidRoPE
from ..nn.norm import RMSNorm
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional, Literal

import matplotlib.pyplot as plt
import math

def modulate(x, shift, scale):
    # TODO: make sure this also makes sense for the single-frame-level transformer
    # shift for mult frames is batch dur dim
    # shift for single frame is batch*dur 1 dim --> should work.
    b, s, d = x.shape
    toks_per_frame = s // shift.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]
    x = x.reshape(b, s, d)
    return x

def gate(x, gate):
    b, s, d = x.shape
    toks_per_frame = s // gate.shape[1]
    x = x.reshape(b, -1, toks_per_frame, d)
    x = x * gate[:, :, None, :]
    x = x.reshape(b, s, d)
    return x


class SeparableCausalDiT(nn.Module):
    # inspired by https://arxiv.org/abs/2602.10095
    def __init__(self, height, width, n_window, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks_encoder=6, n_blocks_decoder=3, 
                       n_registers=1, n_actions=4, bidirectional=False, 
                       debug=False, 
                       rope_C=10000,
                       rope_tmax=None,
                       rope_type: Literal["rope", "learn", "vid"] = "rope",
                       use_flex: bool = False):
        super().__init__()
        self.height = height
        self.width = width
        self.n_window = n_window
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.expansion = expansion
        self.n_registers = n_registers
        self.T = T
        self.patch_size = patch_size
        self.debug = debug
        self.bidirectional = bidirectional
        self.toks_per_frame = (height//patch_size)*(width//patch_size) + n_registers
        self.rope_C = rope_C
        self.use_flex = use_flex
        self.encoder = VideoEncoder(height=height, width=width, n_window=n_window, d_model=d_model, T=T, in_channels=in_channels, patch_size=patch_size, n_heads=n_heads, expansion=expansion, n_blocks=n_blocks_encoder, n_registers=n_registers, n_actions=n_actions, bidirectional=bidirectional, debug=debug, rope_C=rope_C, rope_tmax=rope_tmax, rope_type=rope_type, use_flex=use_flex)
        self.decoder = FrameDecoder(height=height, width=width, d_model=d_model, T=T, in_channels=in_channels, patch_size=patch_size, n_heads=n_heads, expansion=expansion, n_blocks=n_blocks_decoder, n_registers=n_registers, debug=debug, rope_C=rope_C, rope_tmax=rope_tmax, rope_type=rope_type, use_flex=use_flex)
    
    def forward(self, 
                z_for_encoder: Float[Tensor, "batch dur channels height width"], 
                z_for_decoder: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"],
                cached_k: Optional[Float[Tensor, "layer batch dur seq d"]] = None,
                cached_v: Optional[Float[Tensor, "layer batch dur seq d"]] = None):
        # TODO: consider sharing patchify between encoder and decoder
        cond_from_encoder, cached_k, cached_v = self.encoder(z_for_encoder, actions, ts, cached_k=cached_k, cached_v=cached_v)
        out = self.decoder(z_for_decoder, cond_from_encoder, ts)
        return out, cached_k, cached_v
    
    def create_cache(self, batch_size):
        return self.encoder.create_cache(batch_size)

    def create_cache2(self, batch_size):
        return self.encoder.create_cache2(batch_size)

    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype

class FrameDecoder(nn.Module):
    # TODO: actions: yes, no, maybe??
    def __init__(self, height, width, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, 
                       debug=False, 
                       rope_C=10000,
                       rope_tmax=None,
                       rope_type="rope",
                       use_flex: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.expansion = expansion
        self.n_blocks = n_blocks
        self.rope_C = rope_C
        if rope_tmax is None:
            self.rope_tmax = (height//patch_size) * (width//patch_size) + n_registers
        else:
            self.rope_tmax = rope_tmax
        self.rope_type = rope_type
        self.use_flex = use_flex
        self.n_registers = n_registers
        self.T = T 
        self.transformer = Transformer(n_blocks=n_blocks, 
                                       d_model=d_model, 
                                       expansion=expansion, 
                                       n_heads=n_heads, 
                                       use_flex=use_flex, 
                                       rope_tmax=self.rope_tmax, 
                                       rope_C=rope_C)
        self.patch = Patch(in_channels=in_channels, out_channels=d_model, patch_size=patch_size)
        self.norm = RMSNorm(d_model)
        self.unpatch = UnPatch(height, width, in_channels=d_model, out_channels=in_channels, patch_size=patch_size)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.time_emb = NumericEncoding(dim=d_model, n_max=T)
        self.time_emb_mixer = nn.Linear(d_model, d_model)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
    
    def forward(self,
                z_t: Float[Tensor, "batch dur channels height width"], 
                cond_from_encoder: Float[Tensor, "batch dur seq d"],
                ts: Int[Tensor, "batch dur"]):
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z_t.shape[1])

        ts_scaled = (ts.float() * (self.T - 1)).long()
        cond_t = self.time_emb_mixer(self.time_emb(ts_scaled)) # this conditioning will enter via the adaln
        z = self.patch(z_t) # batch dur seq d
        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        print(zr.shape, cond_from_encoder.shape)
        zr += cond_from_encoder
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch * durzr, -1, d) # batchdur seq d
        cond_t = cond_t.reshape(batch * durzr, 1, d) # ok i guess here is different shape
        zr, _, _ = self.transformer(zr, cond_t)
        # unpatchify...
        mu, sigma = self.modulation(cond_t).chunk(2, dim=-1)
        zr = modulate(self.norm(zr), mu, sigma)
        zr = zr.reshape(batch, durzr, seqzr, d)
        out = self.unpatch(zr[:, :, :-self.n_registers])
        return out
    
    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype


class VideoEncoder(nn.Module):
    def __init__(self, height, width, n_window, d_model, T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, 
                       n_registers=1, n_actions=4, bidirectional=False, 
                       debug=False, 
                       rope_C=10000,
                       rope_tmax=None,
                       rope_type: Literal["rope", "learn", "vid"] = "rope",
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
        self.rope_tmax = self.n_window*self.toks_per_frame if rope_tmax is None else rope_tmax
        self.transformer = Transformer(n_blocks, d_model, expansion, n_heads, use_flex, self.rope_tmax, rope_C)
        self.patch = Patch(in_channels=in_channels, out_channels=d_model, patch_size=patch_size)
        self.norm = RMSNorm(d_model)
        self.unpatch = UnPatch(height, width, in_channels=d_model, out_channels=in_channels, patch_size=patch_size)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.registers = nn.Parameter(t.randn(n_registers, d_model) * 1/d_model**0.5)
        self.time_emb = NumericEncoding(dim=d_model, n_max=T)
        self.time_emb_mixer = nn.Linear(d_model, d_model)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        if not self.use_flex:
            if not self.bidirectional:
                self.register_buffer("mask", self.causal_mask())
            else:
                self.register_buffer("mask", None)
        else:
            if not self.bidirectional:
                self.mask = self.causal_mask()
            else:
                self.mask = None
    
    def forward(self, 
                z: Float[Tensor, "batch dur channels height width"], 
                actions: Float[Tensor, "batch dur"],
                ts: Int[Tensor, "batch dur"],
                cached_k: Optional[Float[Tensor, "layer batch dur seq d"]] = None,
                cached_v: Optional[Float[Tensor, "layer batch dur seq d"]] = None):
        if ts.shape[1] == 1:
            ts = ts.repeat(1, z.shape[1])

        a = self.action_emb(actions) # batch dur d
        ts_scaled = (ts.float() * (self.T - 1)).long()
        cond = self.time_emb_mixer(self.time_emb(ts_scaled)) + a
        z = self.patch(z) # batch dur seq d

        # self.registers is in 1x
        zr = t.cat((z, self.registers[None, None].repeat([z.shape[0], z.shape[1], 1, 1])), dim=2)# z plus registers
        if self.bidirectional or cached_k is not None:
            mask_self = None
        else:
            mask_self = self.mask
        batch, durzr, seqzr, d = zr.shape
        zr = zr.reshape(batch, -1, d) # batch durseq d        
        zr, k_update, v_update = self.transformer(zr, cond, mask_self, cached_k=cached_k, cached_v=cached_v)
        return zr.reshape(batch, durzr, seqzr, d), k_update, v_update
    
    def causal_mask(self):
        if self.use_flex:
            return create_block_mask(create_block_causal_mask_mod(self.toks_per_frame), B=None, H=None, Q_LEN=self.toks_per_frame*self.n_window, KV_LEN=self.toks_per_frame*self.n_window)
        else:
            size = self.n_window
            m_self = t.tril(t.ones((size, size), dtype=t.int8, device=self.device)) # - t.tril(t.ones((size, size), dtype=t.int8, device=self.device), diagonal=-self.n_window) # this would be useful if we go bigger than windowxwindow
            m_self = t.kron(m_self, t.ones((self.toks_per_frame, self.toks_per_frame), dtype=t.int8, device=self.device))
            m_self = m_self.to(bool)
            return m_self
    
    def create_cache(self, batch_size):
        return KVCache(batch_size, self.n_blocks, self.n_heads, self.d_head, self.toks_per_frame, self.n_window, dtype=self.dtype, device=self.device)

    def create_cache2(self, batch_size):
        return KVCacheNaive(batch_size, self.n_blocks, self.n_heads, self.d_head, self.toks_per_frame, self.n_window, dtype=self.dtype, device=self.device)

    @property
    def device(self):
        return self.parameters().__next__().device
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx, d_model, expansion, n_heads, rope=None, use_flex=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.expansion = expansion
        self.n_heads = n_heads
        self.norm1 = RMSNorm(d_model)
        self.selfattn = Attention(d_model, n_heads, rope=rope, use_flex=use_flex)
        self.norm2 = RMSNorm(d_model)
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
        z, k_new, v_new = self.selfattn(z, mask=mask_self, k_cache=cached_k, v_cache=cached_v)            
        z = residual + gate(z, c1)

        residual = z
        z = modulate(self.norm2(z), mu2, sigma2)
        z = self.geglu(z)
        z = residual + gate(z, c2)
        return z, k_new, v_new
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype
    
    @property
    def device(self):
        return self.parameters().__next__().device

class Transformer(nn.Module):
    def __init__(self, n_blocks, d_model, expansion, n_heads, use_flex, rope_tmax, rope_C):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads

        self.rope_seq = RoPE(d_model//n_heads, rope_tmax, C=rope_C)
        self.blocks = nn.ModuleList([TransformerBlock(lidx, d_model, expansion, n_heads, rope=self.rope_seq, use_flex=use_flex) for lidx in range(n_blocks)])
    
    def forward(self, z, cond, mask=None, cached_k=None, cached_v=None):
        k_update = []
        v_update = []
        for bidx, block in enumerate(self.blocks):
            ks = cached_k[bidx] if cached_k is not None else None 
            vs = cached_v[bidx] if cached_v is not None else None
            z, k_new, v_new = block(z, cond, mask, cached_k=ks, cached_v=vs)
            if k_new is not None:
                k_update.append(k_new.unsqueeze(0))
                v_update.append(v_new.unsqueeze(0))
        if len(k_update) > 0:
            k_update = t.cat(k_update, dim=0)
            v_update = t.cat(v_update, dim=0)
        return z, k_update, v_update

    @property
    def dtype(self):
        return self.parameters().__next__().dtype
    
    @property
    def device(self):
        return self.parameters().__next__().device


def get_model(height, width, 
              n_window=5, 
              d_model=64, 
              T=100, 
              n_blocks_encoder=6,
              n_blocks_decoder=3,
              patch_size=2, 
              n_heads=8, 
              bidirectional=False, 
              in_channels=3, 
              C=10000, 
              rope_type: Literal["rope", "learn", "vid"] = "rope",
              use_flex=False):
    if rope_type != "rope":
        raise ValueError(f"Rope type {rope_type} not supported for SeparableCausalDiT")
    return SeparableCausalDiT(height=height, 
                              width=width, 
                              n_window=n_window,
                              d_model=d_model, 
                              T=T, 
                              in_channels=in_channels, 
                              patch_size=patch_size, 
                              n_heads=n_heads, 
                              n_blocks_encoder=n_blocks_encoder,
                              n_blocks_decoder=n_blocks_decoder,    
                              bidirectional=bidirectional, 
                              rope_C=C, 
                              rope_type=rope_type, 
                              use_flex=use_flex)


if __name__ == "__main__":
    print("running w/o cache")
    dit = SeparableCausalDiT(height=20, width=20, n_window=25, d_model=64, T=5, n_blocks_encoder=2, n_blocks_decoder=2, rope_C=5000)
    z = t.rand((2, 6, 3, 20, 20))
    actions = t.randint(4, (2, 6))
    ts = t.rand((2, 6))
    out, _, _ = dit(z_for_encoder=z, z_for_decoder=z, actions=actions, ts=ts)
    print(z.shape)
    print(out.shape)
