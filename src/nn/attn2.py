import torch as t
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from functools import partial

from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def causal_mod(score, b, h, q_idx, kv_idx):
    return t.where(q_idx >= kv_idx, score, float("-inf"))

def create_block_causal_mask_mod(block_size):
    def block_causal_mask_mod(b, h, q_idx, kv_idx):
        # either q is in a later block or q and k are in the same block
        return ((q_idx >= kv_idx) | ((q_idx // block_size) == (kv_idx // block_size)))
    return block_causal_mask_mod

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, use_flex=True, rope=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by d_head"

        self.QKV = nn.Linear(self.d_model, 3 * self.d_model)
        self.O = nn.Linear(self.d_model, self.d_model)
        self.lnq = nn.RMSNorm(self.d_head)
        self.lnk = nn.RMSNorm(self.d_head)
        self.rope = rope
        self.use_flex = use_flex

    def forward(self, x, mask=None, k_cache=None, v_cache=None):
        # x: batch x seq x d_model
        if k_cache is None and v_cache is None:
            qkv = self.QKV(x)
            q, k, v = qkv.chunk(3, dim=-1)
            b, s, d = q.shape
            q = q.reshape(b, s, self.n_heads, self.d_head)
            k = k.reshape(b, s, self.n_heads, self.d_head)
            v = v.reshape(b, s, self.n_heads, self.d_head)
            k_new = k 
            v_new = v 
            offset = 0
        else:
            qkv = self.QKV(x)
            q, k_new, v_new = qkv.chunk(3, dim=-1)
            b, s, d = q.shape
            q = q.reshape(b, s, self.n_heads, self.d_head)
            k_new = k_new.reshape(b, s, self.n_heads, self.d_head)
            v_new = v_new.reshape(b, s, self.n_heads, self.d_head)
            k = t.cat([k_cache, k_new], dim=1)
            v = t.cat([v_cache, v_new], dim=1)
            offset = k_cache.shape[1]
        q = self.lnq(q).to(dtype=self.dtype)
        k = self.lnk(k).to(dtype=self.dtype)
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k)
        if self.use_flex:
            print("using flex attention")
            # flex attn expects batch x nhead x seq x dhead
            q_flex = q.permute(0, 2, 1, 3)
            k_flex = k.permute(0, 2, 1, 3)
            v_flex = v.permute(0, 2, 1, 3)
            if mask is not None:
                z = flex_attention(q_flex, k_flex, v_flex, scale=1., block_mask = mask)
            else:
                z = flex_attention(q_flex, k_flex, v_flex, scale=1.)
            z = z.permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            # q, k, v: (batch, n_heads, seq, d_head)
            attn = (q @ k.permute(0, 1, 3, 2)) # batch x nh x seqq x seqk
            if mask is not None and k_cache is None:
                print("applyign mask")
                attn = t.where(mask[:attn.shape[-2], :attn.shape[-1]], attn, float("-inf"))
            probas = attn.softmax(dim=-1)
            z = probas @ v
            # z ... batch x nh x seq x dh
            z = z.permute(0, 2, 1, 3)
            # z ... batch x seq x nh x dh
        z = self.O(z.reshape(b, s, d))
        return z, k_new, v_new

    def mask(self, s: int):
        return t.tril(t.ones((s, s), dtype=bool, device=self.device))

    @property
    def device(self):
        return self.QKV.weight.device

    @property
    def dtype(self):
        return self.QKV.weight.dtype

if __name__ == "__main__":
    t.set_float32_matmul_precision("high")
    t.manual_seed(0)
    import time

    # No device or dtype restriction for flex_attention (assuming it supports both CUDA and CPU)
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    dtype = t.bfloat16

    x = t.rand(32, 65*30, 384, device=device, dtype=dtype)
    attn = Attention(384, 6, use_flex=False).to(device).to(dtype)
    attn_flex = Attention(384, 6, use_flex=True).to(device).to(dtype)
    def causal_mask(self):
        size = self.n_window
        m_self = t.tril(t.ones((size, size), dtype=t.int8, device=self.device)) # - t.tril(t.ones((size, size), dtype=t.int8, device=self.device), diagonal=-self.n_window) # this would be useful if we go bigger than windowxwindow
        m_self = t.kron(m_self, t.ones((self.toks_per_frame, self.toks_per_frame), dtype=t.int8, device=self.device))
        m_self = m_self.to(bool)
        return ~ m_self
    block_mask = t.tril(t.ones((30, 30), dtype=t.int8, device=device))
    block_mask = t.kron(block_mask, t.ones((65, 65), dtype=t.int8, device=device))
    block_mask = block_mask.to(bool)
    block_mask_flex = create_block_mask(create_block_causal_mask_mod(65), B=None, H=None, Q_LEN=x.shape[1], KV_LEN=x.shape[1])
    
    block_mask_ = create_block_causal_mask_mod(65)
    block_mask_test = t.zeros((65*5, 65*5), dtype=bool, device=device)
    for i in range(65*5):
        for j in range(65*5):
            block_mask_test[i, j] = block_mask_(None, None,i, j)
    plt.imshow(block_mask_test.cpu().detach().numpy())
    plt.savefig("block_mask_test.png")
    plt.show()
    assert (block_mask_test == block_mask[:65*5, :65*5]).all()
    # Compile both for fair comparison - CRITICAL for flex attention performance!
    attn_flex.load_state_dict(attn.state_dict())
    
    attn = t.compile(attn)
    attn_flex = t.compile(attn_flex)

    n_rep = 100
    
    # Warmup to trigger compilation
    print("Warming up (compiling)...")

    for idx in range(3):
        _ = attn(x, mask = block_mask)
    if device == "cuda":
        t.cuda.synchronize()
    
    print("Benchmarking vanilla attention...")

    if device == "cuda":
        t.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_rep):
        y_ref, k_cache, v_cache = attn(x, mask = block_mask)
    if device == "cuda":
        t.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"{n_rep} x Vanilla Attention forward pass took {elapsed:.6f} seconds")

    # Warmup flex attention
    print("Warming up flex attention (compiling)...")
    for _ in range(3):
        _ = attn_flex(x, mask = block_mask_flex)
    if device == "cuda":
        t.cuda.synchronize()
    
    print("Benchmarking flex attention...")
    if device == "cuda":
        t.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_rep):
        y_flex, k_cache, v_cache = attn_flex(x, mask = block_mask_flex)
    if device == "cuda":
        t.cuda.synchronize()
    elapsed = time.time() - start_time
    print(f"{n_rep} x Flex Attention forward pass took {elapsed:.6f} seconds")
    loss = y_flex.sum()
    loss.backward()
    print(f"Max absolute difference: {t.abs(y_ref - y_flex).max().item()}")
    print(f"Mean absolute difference: {t.abs(y_ref - y_flex).mean().item()}")
    print("Outputs close (atol=1e-2):", t.allclose(y_ref, y_flex, atol=1e-2, rtol=1e-2))
    print(f"Input shape: {x.shape}, Output shape: {y_ref.shape}")