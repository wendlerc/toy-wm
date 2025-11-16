# Flex Attention Performance Guide

## TL;DR

**For simple causal attention: DON'T use flex_attention. Use vanilla implementation + torch.compile.**

Flex attention is 2-3x **slower** for simple masking patterns like causal attention.

## Benchmark Results

Testing on: batch=32, seq=1950, d_model=384, heads=6, causal masking

| Implementation | Time | Speedup | Notes |
|---|---|---|---|
| **Vanilla + compile** | **6.74s** | **1.00x (fastest)** | ✅ **Recommended for simple causal** |
| Vanilla (no compile) | 10.61s | 0.64x | |
| Flex (no compile) | 18.63s | 0.36x | ⚠️ 2.76x slower |
| Flex + compile | 21.27s | 0.32x | ⚠️ 3.15x slower (worst) |

## Why is Flex Attention Slower?

1. **Overhead for Flexibility**: Flex attention adds abstraction layers that enable flexible masking patterns. For simple patterns, this is pure overhead.

2. **Vanilla + Compile Uses Optimal Kernels**: `torch.compile()` recognizes the simple causal pattern and uses highly optimized cuBLAS/cuDNN kernels with kernel fusion.

3. **Compilation Conflicts**: Wrapping flex_attention with `torch.compile()` on the module level interferes with flex attention's internal compilation, making it even slower.

## When to Use Flex Attention

Flex attention is beneficial for:

### ✅ Complex Masking Patterns
- **Sliding window attention**: Only attend to a local window
- **Block-sparse attention**: Structured sparsity patterns
- **Custom attention patterns**: e.g., document-level boundaries, hierarchical structures

### ✅ Memory-Constrained Scenarios
- Very long sequences (>4096 tokens)
- Doesn't materialize the full NxN attention matrix
- Memory savings can outweigh speed penalty

### ✅ Dynamic/Conditional Masking
- Masking patterns that change based on input
- Pattern depends on runtime conditions

### ✅ Research & Prototyping
- Rapid experimentation with novel attention patterns
- Easy to implement complex masking logic

## When to Use Vanilla Attention

Use vanilla implementation with `torch.compile()` for:

### ✅ Simple Patterns (Recommended)
- **Causal/autoregressive attention** (your case)
- Full attention (no masking)
- Simple fixed masks

### ✅ Maximum Performance
- When speed is critical
- Production deployments with standard patterns

## Example: Sliding Window Attention (Where Flex Wins)

```python
from torch.nn.attention.flex_attention import flex_attention

def sliding_window_causal(score, b, h, q_idx, kv_idx, window=256):
    """Causal + sliding window: only attend to last 256 tokens"""
    causal = q_idx >= kv_idx
    window_mask = (q_idx - kv_idx) <= window
    return torch.where(causal & window_mask, score, float("-inf"))

# For very long sequences, this saves memory and can be faster
# than materializing the full attention matrix
z = flex_attention(q, k, v, score_mod=sliding_window_causal)
```

For this pattern, vanilla attention would need to:
1. Compute full NxN attention matrix (memory intensive)
2. Apply sliding window mask
3. Apply causal mask

Flex attention computes only the non-masked positions, saving memory and potentially time for very long sequences.

## Recommendations for Your Code

### For Simple Causal Attention (Current Use Case)

```python
# RECOMMENDED: Use vanilla implementation
class Attention(nn.Module):
    def forward(self, x):
        # ... compute q, k, v ...
        
        # Vanilla implementation (fast with torch.compile)
        q = q.permute(0, 2, 1, 3)  # (B, H, S, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.tril(torch.ones(S, S, device=x.device))
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        out = F.softmax(attn, dim=-1) @ v
        # ... output projection ...

# Compile the module
model = torch.compile(Attention(...))
```

### If You Need Complex Masking

```python
# Only use flex_attention if you need complex patterns
from torch.nn.attention.flex_attention import flex_attention

# Define score_mod at module level (not in forward())
def my_complex_pattern(score, b, h, q_idx, kv_idx):
    # Your complex logic here
    return score

class FlexAttention(nn.Module):
    def forward(self, x):
        # ... compute q, k, v ...
        
        # Use flex_attention WITHOUT compiling the module
        out = flex_attention(q, k, v, score_mod=my_complex_pattern)
        # ... output projection ...

# DON'T compile the module if using flex_attention
model = FlexAttention(...)
```

## Key Takeaways

1. **Flex attention is NOT a drop-in replacement for standard attention**
2. **It trades speed for flexibility** - only worth it for complex patterns
3. **For causal attention: stick with vanilla + torch.compile**
4. **Only switch to flex when you need its unique capabilities**

## Related Resources

- [PyTorch Flex Attention Docs](https://pytorch.org/docs/main/generated/torch.nn.attention.flex_attention.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - for memory-efficient attention
- For simple patterns, consider `F.scaled_dot_product_attention()` which auto-selects optimal kernel

