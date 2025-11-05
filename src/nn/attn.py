from torch import nn
import torch as t
import einops 
from jaxtyping import Float, Bool
from torch import Tensor
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention


class Attention(nn.Module):
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
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32))
        self.rope = rope


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
            # only compute new keys, values, and rows/cols in attention matrix, and compute only the new output
            q = einops.einsum(x[:,-1].unsqueeze(1), self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k_new = einops.einsum(x[:,-1].unsqueeze(1), self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v_new = einops.einsum(x[:,-1].unsqueeze(1), self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            k = t.cat([k_cache, k_new],dim=1)
            v = t.cat([v_cache, v_new],dim=1)
        else:
            # compute everything from scratch
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
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
        
        print(f"q_perm {q_perm.shape}, k_perm {k_perm.shape}, v_perm {v_perm.shape}")
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
        print(f"z {z.shape}")
        z = z.permute(0, 2, 1, 3)  # Back to (batch, posq, n_heads, d_head)
        out = einops.einsum(z, self.W_O, 'b s n h, n h d -> b s n d')
        out = out.sum(dim=2) + self.b_O
        #print(f"out {out.shape}, attention {probas.shape}, q {q.shape}, k {k.shape}, v {v.shape}")
        return out, z, None

class AttentionSlow(nn.Module):
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
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32))
        self.rope = rope


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
            # only compute new keys, values, and rows/cols in attention matrix, and compute only the new output
            q = einops.einsum(x[:,-1].unsqueeze(1), self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k_new = einops.einsum(x[:,-1].unsqueeze(1), self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v_new = einops.einsum(x[:,-1].unsqueeze(1), self.W_V, 'b s d, n d h -> b s n h') + self.b_V
            k = t.cat([k_cache, k_new],dim=1)
            v = t.cat([v_cache, v_new],dim=1)
        else:
            # compute everything from scratch
            q = einops.einsum(x_q, self.W_Q, 'b s d, n d h -> b s n h') + self.b_Q
            k = einops.einsum(x_kv, self.W_K, 'b s d, n d h -> b s n h') + self.b_K
            v = einops.einsum(x_kv, self.W_V, 'b s d, n d h -> b s n h') + self.b_V
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        attention = einops.einsum(q, k, 'b sq n h, b sk n h -> b n sq sk')
        attention /= d_head**0.5 # this was the mistake :D:D:D I had d_model here initially
        if mask is not None:
            attention = t.where(mask, self.IGNORE, attention)
        probas = attention.softmax(dim=3)
        z = einops.einsum(probas, v, 'b n sq sk, b sk n h -> b sq n h')
        out = einops.einsum(z, self.W_O, 'b s n h, n h d -> b s n d')
        out = out.sum(dim=2) + self.b_O
        #print(f"out {out.shape}, attention {probas.shape}, q {q.shape}, k {k.shape}, v {v.shape}")
        return out, z, None


if __name__ == "__main__":
    attn_slow = AttentionSlow(d_model=256, n_heads=8)
    attn = Attention(d_model=256, n_heads=8)
    attn.load_state_dict(attn_slow.state_dict())
    attn.double()
    attn_slow.double()
    x = t.randn(1, 1000, 256, dtype=t.float64)*10
    mask = t.randint(0, 2, (1000, 1000), dtype=t.bool)
    y, z, _ = attn(x, x, mask=mask)
    y_slow, z_slow, _ = attn_slow(x, x, mask=mask)
    assert t.allclose(z, z_slow, atol=1e-5), f"Attention and AttentionSlow should be the same: {(z - z_slow).abs().max()}"
    assert t.allclose(y, y_slow, atol=1e-5), f"Attention and AttentionSlow should be the same: {(y - y_slow).abs().max()}"
    print("Attention and AttentionSlow are the same")
