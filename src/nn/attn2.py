import torch as t
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def causal_mod(score, b, h, q_idx, kv_idx):
    return t.where(q_idx >= kv_idx, score, float("-inf"))

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, debug=False, use_flex=True, rope=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.debug = debug
        assert d_model % n_heads == 0, "d_model must be divisible by d_head"

        self.QKV = nn.Linear(self.d_model, 3 * self.d_model)
        self.O = nn.Linear(self.d_model, self.d_model)
        self.lnq = nn.LayerNorm(self.d_head)
        self.lnk = nn.LayerNorm(self.d_head)
        self.rope = rope
        self.use_flex = use_flex
        self.scale = 1.0 / (self.d_head ** 0.5)

    def forward(self, x):
        # x: batch x seq x d_model
        qkv = self.QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, s, d = q.shape
        q = q.reshape(b, s, self.n_heads, self.d_head)
        k = k.reshape(b, s, self.n_heads, self.d_head)
        v = v.reshape(b, s, self.n_heads, self.d_head)
        q = self.lnq(q)
        k = self.lnk(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        if self.use_flex:
            # flex attn expects batch x nhead x seq x dhead
            q_flex = q.permute(0, 2, 1, 3)
            k_flex = k.permute(0, 2, 1, 3)
            v_flex = v.permute(0, 2, 1, 3)
            #q_flex = t.nested.nested_tensor(q_flex, layout=t.jagged)
            #k_flex = t.nested.nested_tensor(k_flex, layout=t.jagged)
            #v_flex = t.nested.nested_tensor(v_flex, layout=t.jagged)
            if self.causal:
                z = flex_attention(q_flex, k_flex, v_flex, 
                                  score_mod=causal_mod, scale=self.scale)
            else:
                z = flex_attention(q_flex, k_flex, v_flex, scale=self.scale)
            z = z.permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            # q, k, v: (batch, n_heads, seq, d_head)
            attn = (q @ k.permute(0, 1, 3, 2)) * self.scale  # batch x nh x seqq x seqk
            if self.causal:
                attn = t.where(self.mask(s), attn, float("-inf"))
            probas = attn.softmax(dim=-1)
            if self.debug:
                plt.imshow(probas[0, 0].cpu().detach().numpy())
                plt.show()
            z = probas @ v
            # z ... batch x nh x seq x dh
            z = z.permute(0, 2, 1, 3)
            # z ... batch x seq x nh x dh
        z = self.O(z.reshape(b, s, d))
        return z

    def mask(self, s: int):
        return t.tril(t.ones((s, s), dtype=bool, device=self.device))

    @property
    def device(self):
        return self.QKV.weight.device

    @property
    def dtype(self):
        return self.QKV.weight.dtype

if __name__ == "__main__":
    import time
    t.manual_seed(0)
    
    # No device or dtype restriction for flex_attention (assuming it supports both CUDA and CPU)
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    dtype = t.bfloat16

    attn = Attention(384, 6, use_flex=False, causal=True).to(device).to(dtype)
    attn_flex = Attention(384, 6, use_flex=True, causal=True).to(device).to(dtype)
    
    # Compile both for fair comparison - CRITICAL for flex attention performance!
    attn_flex.load_state_dict(attn.state_dict())
    attn_flex.eval()
    attn.eval()
    
    attn = t.compile(attn)
    attn_flex = t.compile(attn_flex)

    x = t.rand(32, 65*30, 384, device=device, dtype=dtype)
    
    # Warmup to trigger compilation
    print("Warming up (compiling)...")
    with t.no_grad():
        for _ in range(3):
            _ = attn(x)
        if device == "cuda":
            t.cuda.synchronize()
    
    print("Benchmarking vanilla attention...")
    with t.no_grad():
        if device == "cuda":
            t.cuda.synchronize()
        start_time = time.time()
        for _ in range(500):
            y_ref = attn(x)
        if device == "cuda":
            t.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"Vanilla Attention forward pass took {elapsed:.6f} seconds")
    
    # Warmup flex attention
    print("Warming up flex attention (compiling)...")
    with t.no_grad():
        for _ in range(3):
            _ = attn_flex(x)
        if device == "cuda":
            t.cuda.synchronize()
    
    print("Benchmarking flex attention...")
    with t.no_grad():
        if device == "cuda":
            t.cuda.synchronize()
        start_time = time.time()
        for _ in range(500):
            y_flex = attn_flex(x)
        if device == "cuda":
            t.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"Flex Attention forward pass took {elapsed:.6f} seconds")
    
    print(f"Max absolute difference: {t.abs(y_ref - y_flex).max().item()}")
    print(f"Mean absolute difference: {t.abs(y_ref - y_flex).mean().item()}")
    print("Outputs close (atol=1e-2):", t.allclose(y_ref, y_flex, atol=1e-2, rtol=1e-2))
    print(f"Input shape: {x.shape}, Output shape: {y_ref.shape}")