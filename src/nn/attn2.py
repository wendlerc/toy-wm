import torch as t
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

try:
    from flash_attn import flash_attn_func
    _has_flashattn = True
except ImportError:
    _has_flashattn = False

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, debug=False, use_flash=True, rope=None):
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
        if use_flash and not _has_flashattn:
            raise ImportError("flash-attn is not installed.")
        self.use_flash = use_flash

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
        if self.use_flash:
            # flash_attn_func expects shape (batch, seqlen, nheads, d_head) and fp16/bf16 on CUDA
            q_flash = q.contiguous()
            k_flash = k.contiguous()
            v_flash = v.contiguous()
            # shape: (batch, seq, n_heads, d_head)
            # flash_attn_func(q, k, v, dropout_p, softmax_scale, causal)
            softmax_scale = self.d_head ** (-0.5)
            z = flash_attn_func(q_flash, k_flash, v_flash, 0.0, softmax_scale, self.causal)
            # output: (batch, seq, n_heads, d_head) --> we don't need to permute to that order
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            # q, k, v: (batch, n_heads, seq, d_head)
            attn = q @ k.permute(0, 1, 3, 2) * self.d_head ** (-0.5)  # batch x nh x seqq x seqk
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
    
    # Flash attention requires CUDA
    device = 'cuda' if t.cuda.is_available() and _has_flashattn else 'cpu'
    
    attn = Attention(384, 6, use_flash=False).to(device)
    attn_flash = Attention(384, 6, use_flash=True).to(device) if _has_flashattn and t.cuda.is_available() else None
    
    # Flash attention requires fp16 or bf16
    if _has_flashattn and t.cuda.is_available():
        attn = attn.to(t.bfloat16)
        attn_flash = attn_flash.to(t.bfloat16)
        x = t.rand(64, 65*30, 384, device=device, dtype=t.bfloat16)
    else:
        x = t.rand(64, 65*30, 384, device=device)
    
    with t.no_grad():
        t.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            y_ref = attn(x)
        t.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"Vanilla Attention forward pass took {elapsed:.6f} seconds")
    
    if _has_flashattn and t.cuda.is_available():
        # Copy the weights for equivalence
        attn_flash.load_state_dict(attn.state_dict())
        attn_flash.eval()
        attn.eval()
        
        with t.no_grad():
            t.cuda.synchronize()
            start_time = time.time()
            for _ in range(100):
                y_flash = attn_flash(x)
            t.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"Flash Attention forward pass took {elapsed:.6f} seconds")
        
        print(f"Max absolute difference: {t.abs(y_ref - y_flash).max().item()}")
        print(f"Mean absolute difference: {t.abs(y_ref - y_flash).mean().item()}")
        # Note: Some numerical differences expected due to fp16/bf16 precision and different computation order
        print("Outputs close (atol=1e-2):", t.allclose(y_ref, y_flash, atol=1e-2, rtol=1e-2))
    elif _has_flashattn and not t.cuda.is_available():
        print("flash-attn is installed but CUDA is not available; only vanilla attention tested.")
    else:
        print("flash-attn not installed; only vanilla attention tested.")
    
    print(f"Input shape: {x.shape}, Output shape: {y_ref.shape}")