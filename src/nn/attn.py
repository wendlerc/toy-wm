from torch import nn
from torch.nn import functional as F
import torch as t
import einops 
from jaxtyping import Float, Bool
from torch import Tensor
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention
from matplotlib import pyplot as plt

from pdb import set_trace

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

    
class AttentionEinOps(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, d_model, n_heads, rope=None, ln_first=False):
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model} must be divisble by {n_heads}"
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln_first = ln_first 
        d_head = self.d_head
        self.W_Q = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_K = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_V = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        self.b_Q = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_K = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_V = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_O = nn.Parameter(t.zeros((d_model)))
        nn.init.normal_(self.W_Q, 1/d_model**0.5)
        nn.init.normal_(self.W_K, 1/d_model**0.5)
        nn.init.normal_(self.W_V, 1/d_model**0.5)
        nn.init.normal_(self.W_O, 1/d_head**0.5)
        self.register_buffer("IGNORE", t.tensor(float('-inf'), dtype=t.float32))
        self.rope = rope
        self.ln1 = nn.LayerNorm(d_head)
        self.ln2 = nn.LayerNorm(d_head)


    def forward(
        self, 
        x_q: Float[Tensor, "batch posq d_model"],
        x_kv: Float[Tensor, "batch posk d_model"],
        mask: Bool[Tensor, "posq posk"] = None, # the 1s are removed
        k_cache: Optional[Float[Tensor, "batch posk n_head d_head"]] = None, 
        v_cache: Optional[Float[Tensor, "batch posk n_head d_head"]] = None,
    ) -> Float[Tensor, "batch posq d_model"]:
        assert (k_cache is None and v_cache is None) or (k_cache is not None and v_cache is not None), "k_cache and v_cache go together."
        if k_cache is not None and v_cache is not None:
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k_new = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v_new = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            
            k = t.cat([k_cache, k_new], dim=1)
            v = t.cat([v_cache, v_new], dim=1)
            if self.ln_first:
                q = self.ln1(q)
                k = self.ln2(k)

            if self.rope is not None:
                q = self.rope(q, offset=k_cache.shape[1]) 
                k = self.rope(k, offset=0)

            if not self.ln_first:
                q = self.ln1(q) # ppl usually do this before rope but our best checkpoint has it after rope, so this is for bwd compatibility; but in quick test on singleframe this did not make a big difference
                k = self.ln2(k)
            mask = None
        else:
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            if self.ln_first:
                q = self.ln1(q)
                k = self.ln2(k)

            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)
            
            if not self.ln_first:
                q = self.ln1(q)
                k = self.ln2(k) 
            k_new = k
            v_new = v

        attention = einops.einsum(q, k, 'b sq n h, b sk n h -> b n sq sk')
        if mask is not None and k_cache is not None:
            attention = t.where(mask[k_cache.shape[1]:k_cache.shape[1]+q.shape[1], :k.shape[1]], attention, self.IGNORE)
        elif mask is not None:
            if attention.shape[-1] != mask.shape[-1] or attention.shape[-2] != mask.shape[-2]:
                mask = mask[:attention.shape[-1], :attention.shape[-2]]
            attention = t.where(mask, attention, self.IGNORE) 
        probas = attention.softmax(dim=3)
        z = einops.einsum(probas, v, 'b n sq sk, b sk n h -> b sq n h')
        out = einops.einsum(z, self.W_O, 'b s n h, n h d -> b s n d')
        out = out.sum(dim=2) + self.b_O
        return out, k_new, v_new


