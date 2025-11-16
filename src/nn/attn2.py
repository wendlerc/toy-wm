import torch as t
import torch.nn as nn
from torch.nn import  functional as F
from matplotlib import pyplot as plt

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, debug=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads
        self.causal = causal
        self.debug = debug
        assert d_model % n_heads == 0, "d_model must be divisible by d_head"

        self.QKV = nn.Linear(self.d_model, 3*self.d_model)
        self.O = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        # x: batch x seq x d_model
        q, k, v = self.QKV(x).chunk(3, dim=-1)
        # q ... batch x seq x d_model
        b, s, d = q.shape
        q = q.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.reshape(b, s, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # q ... batch x nh x seq x dh
        attn = q @ k.permute(0, 1, 3, 2)*self.d_head**(-0.5) # batch x nh x seqq x seqk
        if self.causal:
            attn = t.where(self.mask(s), attn, float("-inf"))
        probas = attn.softmax(dim=-1)
        if self.debug:
            plt.imshow(probas[0,0].cpu().detach().numpy())
            plt.show()
        z = probas @ v
        # z ... batch x nh x seq x dh
        z = z.permute(0, 2, 1, 3).reshape(b, s, d)
        z = self.O(z) # why is ths equivalent to upprojecting the heads and adding them up?
        # z input to O: batch x seq x dh1+dh2+dh3... (this is the col that gets multiplied with W), thus the result contains the sum of the up-projections
        return z

    def mask(self, s: int):
        return t.tril(t.ones((s,s), dtype=bool, device=self.device))

    @property
    def device(self):
        return self.QKV.weight.device
    
    @property
    def dtype(self):
        return self.QKV.weight.dtype

if __name__ == "__main__":
    attn = Attention(64, 2)
    x = t.rand(8, 100, 64)
    y = attn(x)
    print(x.shape, y.shape)