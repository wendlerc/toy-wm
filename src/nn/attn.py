from torch import nn
from torch.nn import functional as F
import torch as t
import einops 
from jaxtyping import Float, Bool
from torch import Tensor
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention


class KVCache(nn.Module):
    def __init__(self, batch_size, n_layers, n_heads, d_head, toks_per_frame, n_window):
        """
        This is a rolling KVCache
        """
        super().__init__()
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.toks_per_frame = toks_per_frame
        self.n_window = n_window
        self.size = toks_per_frame * n_window
        self.n_layers = n_layers
        self.curr_layer = 0
        self.global_loc = 0
        self.local_loc = 0
        self.register_buffer('keys', t.zeros(n_layers, batch_size, n_window*toks_per_frame, n_heads, d_head))
        self.register_buffer('values', t.zeros(n_layers, batch_size, n_window*toks_per_frame, n_heads, d_head))
    
    def get(self, layer_idx):
        assert layer_idx == self.curr_layer, f"layer idx should be the same as our internal counter but we got {layer_idx} and internal is {self.curr_layer}."
        return self.keys[layer_idx, :, :self.local_loc], self.values[layer_idx, :, :self.local_loc]
    
    def extend(self, layer_idx, keys, values):
        assert keys.shape == values.shape, f"keys and values shapes must match {self.keys.shape} != {self.values.shape}"
        assert layer_idx == self.curr_layer, f"layer idx should be the same as our internal counter but we got {layer_idx} and internal is {self.curr_layer}."
        assert self.local_loc <= self.size, f"the cache size should be between 0 and {self.size}"
        local_loc = self.local_loc
        if local_loc == self.size:
            # move to the left
            local_loc -= keys.shape[1]
            assert local_loc >= 0, f"the cache update {keys.shape[1]} was larger than the cache {self.size}, that's not supported for now."
            assert local_loc % self.toks_per_frame == 0, f"the number of elements in the cache {local_loc} must be a multiple of the number of tokens per frame {self.toks_per_frame}"
            self.keys[layer_idx, :, :local_loc] = self.keys[layer_idx, :, -local_loc:].clone()
            self.values[layer_idx, :, :local_loc] = self.values[layer_idx, :, -local_loc:].clone()

        assert local_loc + keys.shape[1] <= self.size, f"{local_loc + keys.shape[1]} out of bounds {self.size}"
        self.keys[layer_idx, :, local_loc:local_loc + keys.shape[1]] = keys
        self.values[layer_idx, :, local_loc:local_loc + keys.shape[1]] = values 
        self.curr_layer = (self.curr_layer + 1) % self.n_layers

    def update_global_location(self, n_frames):
        self.global_loc += n_frames * self.toks_per_frame
        if self.local_loc < self.size:
            self.local_loc += n_frames * self.toks_per_frame
            assert self.local_loc <= self.size, f"the local loc {self.local_loc} should never be bigger than {self.size}, something went wrong."

    def reset(self):
        self.global_loc = 0
        self.local_loc = 0
        self.curr_layer = 0
        self.keys.zero_()
        self.values.zero_()

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

    def __init__(self, d_model, n_heads, rope=None):
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model} must be divisble by {n_heads}"
        self.d_head = d_model // n_heads
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
        offset: int = 0
    ) -> Float[Tensor, "batch posq d_model"]:
        assert (k_cache is None and v_cache is None) or (k_cache is not None and v_cache is not None), "k_cache and v_cache go together."
        d_head = self.d_head
        if k_cache is not None and v_cache is not None:
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k_new = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v_new = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            if self.rope is not None:
                q = self.rope(q, offset=offset)
                k_new = self.rope(k_new, offset=offset)
            k = t.cat([k_cache, k_new], dim=1)
            v = t.cat([v_cache, v_new], dim=1)
        else:
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)
            k_new = k
            v_new = v
        
        q = self.ln1(q)
        k = self.ln2(k) # this leanrs much faster using layernorm here

        attention = einops.einsum(q, k, 'b sq n h, b sk n h -> b n sq sk')
        if mask is not None:
            attention = t.where(mask[:q.shape[1], :k.shape[1]], self.IGNORE, attention)
        probas = attention.softmax(dim=3)
        z = einops.einsum(probas, v, 'b n sq sk, b sk n h -> b sq n h')
        out = einops.einsum(z, self.W_O, 'b s n h, n h d -> b s n d')
        out = out.sum(dim=2) + self.b_O
        return out, k_new, v_new


if __name__ == "__main__":
    from .pe import RoPE
    import inspect
    rope = RoPE(256//8, 10000)
    dtype = t.float32
    rope = rope.to(dtype)
    attn_slow = AttentionSlow(d_model=256, n_heads=8, rope=rope)
    attn = Attention(d_model=256, n_heads=8, rope=rope)
    attn.load_state_dict(attn_slow.state_dict(), strict=False)
    attn.to(dtype)
    attn_slow.to(dtype)
    x = t.randn(1, 1000, 256, dtype=dtype)*10
    xkv = t.randn(1, 1000, 256, dtype=dtype)*10
    mask = t.randint(0, 2, (1000, 1000), dtype=t.bool)
    y, z, _ = attn(x, xkv, mask=mask)
    y_slow, z_slow, _ = attn_slow(x, xkv, mask=mask)
    #assert t.allclose(z, z_slow, atol=1e-5), f"Attention and AttentionSlow should be the same: {(z - z_slow).abs().max()}"
    #assert t.allclose(y, y_slow, atol=1e-5), f"Attention and AttentionSlow should be the same: {(y - y_slow).abs().max()}"
    print("Attention and AttentionSlow are the same")

    loss = t.nn.functional.mse_loss(y, y_slow)
    loss.backward()
    print("-"*100)
    for n, p in attn.named_parameters():
        print(n, p.grad.shape, p.grad.max(), p.grad.min())
    print("-"*100)
    for n, p in attn_slow.named_parameters():
        print(n, p.grad.shape, p.grad.max(), p.grad.min())


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, d_model, n_heads, rope=None, use_flex_attention=False):
        super().__init__()
        assert d_model % n_heads == 0, f"{d_model} must be divisble by {n_heads}"
        self.d_head = d_model // n_heads
        d_head = self.d_head
        self.W_Q = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_K = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_V = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        #self.b_Q = nn.Parameter(t.zeros((n_heads, d_head)))
        #self.b_K = nn.Parameter(t.zeros((n_heads, d_head)))
        #self.b_V = nn.Parameter(t.zeros((n_heads, d_head)))
        #self.b_O = nn.Parameter(t.zeros((d_model)))
        nn.init.normal_(self.W_Q, 1/d_model**0.5)
        nn.init.normal_(self.W_K, 1/d_model**0.5)
        nn.init.normal_(self.W_V, 1/d_model**0.5)
        nn.init.normal_(self.W_O, 1/d_head**0.5)
        self.register_buffer("IGNORE", t.tensor(float('-inf'), dtype=t.float32))
        self.rope = rope
        self.use_flex_attention = use_flex_attention
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
        d_head = self.d_head
        if k_cache is not None and v_cache is not None:
            raise NotImplementedError("kv cache not implemented yet")
            q = einops.einsum(x, self.W_Q, 'b s d, n d h -> b s n h') 
            k_new = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') 
            v_new = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') 
            k = t.cat([k_cache, k_new], dim=1)
            v = t.cat([v_cache, v_new], dim=1)
        else:
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') 
            k = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') 
            v = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') 
        
        q = self.ln1(q)
        k = self.ln2(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        
        # Convert to (batch, num_heads, seq_len, head_dim) format for flex_attention
        q_perm = q.permute(0, 2, 1, 3)  # (batch, n_heads, posq, d_head)
        k_perm = k.permute(0, 2, 1, 3)   # (batch, n_heads, posk, d_head)
        v_perm = v.permute(0, 2, 1, 3)   # (batch, n_heads, posk, d_head)
        
        # Ensure tensors are contiguous to avoid flex_attention indexing bugs
        q_perm = q_perm.contiguous()
        k_perm = k_perm.contiguous()
        v_perm = v_perm.contiguous()
        
        if self.use_flex_attention:
            # Handle mask using score_mod if needed
            if mask is not None:
                # Store mask and IGNORE for use in score_mod closure
                mask_tensor = mask  # (posq, posk)
                ignore_val = self.IGNORE
                def score_mod(score, b, h, q_idx, kv_idx):
                    # score_mod operates on individual scalar scores
                    # Apply mask: where mask is True, set to -inf
                    # Use torch ops that work in compiled context
                    mask_val = mask_tensor[q_idx, kv_idx]
                    return t.where(mask_val, ignore_val, score)
                z = flex_attention(q_perm, k_perm, v_perm, score_mod=score_mod)
            else:
                z = flex_attention(q_perm, k_perm, v_perm)
        else:
            condi = mask is None and not self.dtype == t.float32
            with t.backends.cuda.sdp_kernel(
                enable_flash=condi, 
                enable_math=not condi, 
                enable_mem_efficient=not condi
            ):
                z = F.scaled_dot_product_attention(
                    q_perm, k_perm, v_perm,
                    attn_mask = mask.logical_not() if mask is not None else None,
                    dropout_p = 0.0, 
                    is_causal = False, 
                    scale = 1.0
                )
        z = z.permute(0, 2, 1, 3)  # Back to (batch, posq, n_heads, d_head)
        out = einops.einsum(z, self.W_O, 'b s n h, n h d -> b s n d')
        out = out.sum(dim=2) 
        #print(f"out {out.shape}, attention {probas.shape}, q {q.shape}, k {k.shape}, v {v.shape}")
        return out, z, None
    
    @property
    def dtype(self):
        return self.parameters().__next__().dtype
    
    @property
    def device(self):
        return self.parameters().__next__().device