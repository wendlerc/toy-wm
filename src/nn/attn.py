import torch as t
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from functools import partial

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from .norm import RMSNorm

class KVCache(nn.Module):
    """
    Rolling KV cache implemented as a ring buffer.
    - Shapes:
        keys/values per extend(): (batch_size, T, n_heads, d_head)
    - Internal storage:
        (n_layers, batch_size, size, n_heads, d_head) where size = toks_per_frame * n_window
    - Semantics:
        Call `extend(layer_idx, k, v)` once per layer for the *same* frame.
        Call `update_global_location(n_frames)` once after all layers to commit the frame(s).
    """
    def __init__(self, batch_size, n_layers, n_heads, d_head, toks_per_frame, n_window, *, dtype=None, device=None):
        super().__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.toks_per_frame = toks_per_frame
        self.n_window = n_window
        self.size = toks_per_frame * (n_window-1) #toks_per_frame # (toks_per_frame * n_window)

        # Pointers / counters
        self.global_loc = 0                 # total tokens ever committed
        self.local_loc = 0                  # valid tokens in buffer (<= size)
        self._write_ptr = 0                 # ring-buffer write pointer (index of next commit position)

        # Storage
        dtype = dtype if dtype is not None else t.float32
        self.register_buffer('keys',   t.zeros(n_layers, batch_size, self.size, n_heads, d_head, dtype=dtype, device=device))
        self.register_buffer('values', t.zeros(n_layers, batch_size, self.size, n_heads, d_head, dtype=dtype, device=device))


    def get(self):
        """Return (K, V) for given layer in chronological order: shape (B, L, H, D) where L = local_loc."""
        if self.local_loc == 0:
            # return empty views
            empty = self.keys[:, :, :0]
            return empty, empty

        start = (self._write_ptr - self.local_loc) % self.size
        if start + self.local_loc <= self.size:
            # contiguous slice
            k = self.keys[:, :, start:start + self.local_loc]
            v = self.values[:, :, start:start + self.local_loc]
        else:
            # wrap: concatenate two slices to maintain chronological order
            first = self.size - start
            k = t.cat([
                self.keys[:, :, start:self.size],
                self.keys[:, :, 0:(self.local_loc - first)]
            ], dim=2)
            v = t.cat([
                self.values[:, :, start:self.size],
                self.values[:, :, 0:(self.local_loc - first)]
            ], dim=2)
        return k, v

    @t.no_grad()
    def extend(self, keys, values):
        """
        Stage (but do not commit) tokens for the current frame for the given layer.
        Call update_global_location(n_frames) to commit after all layers wrote.
        """
        assert keys.shape == values.shape, f"keys and values shapes must match, got {keys.shape} vs {values.shape}"

        L, B, T, H, D = keys.shape
        assert L == self.n_layers, f"nlayers mismatch: expected {self.n_layers}, got {L}"
        assert B == self.batch_size, f"batch mismatch: expected {self.batch_size}, got {B}"
        assert H == self.n_heads and D == self.d_head, f"heads/d_head mismatch: expected {(self.n_heads, self.d_head)}, got {(H, D)}"
        assert T > 0 and T <= self.size, f"T must be in 1..{self.size}, got {T}"

        if keys.dtype != self.keys.dtype or keys.device != self.keys.device:
            keys = keys.to(dtype=self.keys.dtype, device=self.keys.device)
        if values.dtype != self.values.dtype or values.device != self.values.device:
            values = values.to(dtype=self.values.dtype, device=self.values.device)

        i0 = self._write_ptr
        i1 = (self._write_ptr + T) % self.size
        if i0 < i1:
            self.keys[:, :, i0:i1] = keys
            self.values[:, :, i0:i1] = values
        else:
            # wrap
            split = self.size - i0
            self.keys[:, :, i0:self.size] = keys[:, :, :split]
            self.values[:, :, i0:self.size] = values[:, :, :split]
            self.keys[:, :, 0:i1] = keys[:, :, split:]
            self.values[:, :, 0:i1] = values[:, :, split:]

        self.global_loc += keys.shape[2]
        self.local_loc = min(self.size, self.local_loc + keys.shape[2])
        self._write_ptr = (self._write_ptr + keys.shape[2]) % self.size

    @t.no_grad()
    def reset(self, zero_memory: bool = True):
        self.global_loc = 0
        self.local_loc = 0
        self.curr_layer = 0
        self._write_ptr = 0
        if zero_memory:
            self.keys.zero_()
            self.values.zero_()

    @property
    def local_location(self):
        return self.local_loc

    @property
    def global_location(self):
        return self.global_loc

    @property
    def device(self):
        return self.keys.device

    @property
    def dtype(self):
        return self.keys.dtype



class KVCacheNaive(nn.Module): 
    def __init__(self, batch_size, n_layers, n_heads, d_head, toks_per_frame, n_window, dtype=t.float32, device='cuda'):
        """
        This is a rolling KVCache
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.toks_per_frame = toks_per_frame
        self.n_window = n_window
        self.size = toks_per_frame * (n_window - 1)
        self.n_layers = n_layers
        self.global_loc = 0
        self.local_loc = 0

        self.register_buffer('keys', t.zeros(n_layers, batch_size, self.size, n_heads, d_head, dtype=dtype, device=device))
        self.register_buffer('values', t.zeros(n_layers, batch_size, self.size, n_heads, d_head, dtype=dtype, device=device))
    
    def get(self):
        return self.keys[:, :, :self.local_loc], self.values[:, :, :self.local_loc]
    
    def extend(self, keys, values):
        """
        this should only be called on the last denoising step respectively.
        """
        assert keys.shape == values.shape, f"keys and values shapes must match {self.keys.shape} != {self.values.shape}"
        assert self.local_loc <= self.size, f"the cache size should be between 0 and {self.size}"
        local_loc = self.local_loc
        if local_loc == self.size:
            # move to the left
            local_loc -= keys.shape[2]
            assert local_loc >= 0, f"the cache update {keys.shape[2]} was larger than the cache {self.size}, that's not supported for now."
            assert local_loc % self.toks_per_frame == 0, f"the number of elements in the cache {local_loc} must be a multiple of the number of tokens per frame {self.toks_per_frame}"
            self.keys[:, :, :local_loc] = self.keys[:, :, self.toks_per_frame:local_loc+self.toks_per_frame].clone()
            self.values[:, :, :local_loc] = self.values[:, :, self.toks_per_frame:local_loc+self.toks_per_frame].clone()

        assert local_loc + keys.shape[2] <= self.size, f"{local_loc + keys.shape[2]} out of bounds {self.size}"
        self.keys[:, :, local_loc:local_loc + keys.shape[2]] = keys
        self.values[:, :, local_loc:local_loc + keys.shape[2]] = values 
        self.curr_layer = (self.curr_layer + 1) % self.n_layers

        self.global_loc += keys.shape[2]
        if self.local_loc < self.size:
            self.local_loc += keys.shape[2]
            assert self.local_loc <= self.size, f"the local loc {self.local_loc} should never be bigger than {self.size}, something went wrong."

    def reset(self):
        self.global_loc = 0
        self.local_loc = 0
        self.curr_layer = 0
        self.keys.zero_()
        self.values.zero_()

    @property
    def local_location(self):
        return self.local_loc

    @property
    def global_location(self):
        return self.global_loc

    @property
    def device(self):
        return self.keys.device
    
    @property
    def dtype(self):
        return self.keys.dtype


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
        self.lnq = RMSNorm(self.d_head)
        self.lnk = RMSNorm(self.d_head)
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