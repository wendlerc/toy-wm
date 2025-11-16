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
    def __init__(self, d_model, n_heads, causal=True, debug=False, use_flash=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.debug = debug
        assert d_model % n_heads == 0, "d_model must be divisible by d_head"

        self.QKV = nn.Linear(self.d_model, 3 * self.d_model)
        self.O = nn.Linear(self.d_model, self.d_model)
        if use_flash and not _has_flashattn:
            raise ImportError("flash-attn is not installed.")
        self.use_flash = use_flash

    def forward(self, x):
        # x: batch x seq x d_model
        qkv = self.QKV(x)
        q, k, v = qkv.chunk(3, dim=-1)
        b, s, d = q.shape
        q = q.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # q, k, v: (batch, n_heads, seq, d_head)
        if self.use_flash:
            # flash_attn_func expects shape (batch, seqlen, nheads, d_head) and fp16/fp32 on CUDA
            q_flash = q.permute(0, 2, 1, 3).contiguous().to(dtype=x.dtype)
            k_flash = k.permute(0, 2, 1, 3).contiguous().to(dtype=x.dtype)
            v_flash = v.permute(0, 2, 1, 3).contiguous().to(dtype=x.dtype)
            # shape: (batch, seq, n_heads, d_head)
            # flash_attn_func(q, k, v, dropout_p, causal, softmax_scale)
            y = flash_attn_func(q_flash, k_flash, v_flash, 0.0, self.causal, None)
            # output: (batch, seq, n_heads, d_head)
            z = y.permute(0, 2, 1, 3).reshape(b, s, d)
            z = self.O(z)
            return z
        else:
            attn = q @ k.permute(0, 1, 3, 2) * self.d_head ** (-0.5)  # batch x nh x seqq x seqk
            if self.causal:
                attn = t.where(self.mask(s), attn, float("-inf"))
            probas = attn.softmax(dim=-1)
            if self.debug:
                plt.imshow(probas[0, 0].cpu().detach().numpy())
                plt.show()
            z = probas @ v
            # z ... batch x nh x seq x dh
            z = z.permute(0, 2, 1, 3).reshape(b, s, d)
            z = self.O(z)
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
    t.manual_seed(0)
    attn = Attention(64, 2, use_flash=False)
    attn_flash = Attention(64, 2, use_flash=True) if _has_flashattn else None
    x = t.rand(8, 100, 64).to(attn.QKV.weight.device)
    y_ref = attn(x)
    if _has_flashattn:
        # Move Attention with flash to same device and dtype as normal
        attn_flash.to(attn.device).eval()
        # Copy the weights for true equivalence
        attn_flash.load_state_dict(attn.state_dict())
        y_flash = attn_flash(x)
        print(f"Max absolute difference (output): {t.abs(y_ref - y_flash).max().item()}")
        print("Are outputs close?", t.allclose(y_ref, y_flash, atol=1e-4, rtol=1e-4))
    else:
        print("flash-attn not installed; only vanilla attention tested.")
    print(x.shape, y_ref.shape)