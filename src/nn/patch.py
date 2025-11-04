from torch import nn
from einops import rearrange
import torch as t

class Patch(nn.Module): # adapted from https://github.com/cloneofsimo/minRF
    def __init__(self, in_channels=3, out_channels=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        dim = out_channels
        if dim % 32 == 0 and dim > 32:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, dim // 2),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, dim // 2),
            )
        else:
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
            )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

    def forward(self, x):
        batch, dur, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        x = self.init_conv_seq(x)
        x = self.patchify(x)
        x = self.x_embedder(x)
        x = x.reshape(batch, dur, -1, self.out_channels)
        return x

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

class UnPatch(nn.Module):
    def __init__(self, height, width, in_channels=64, out_channels=3, patch_size=2):
        super().__init__()
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unpatch = nn.Linear(in_channels, out_channels*patch_size**2)

    def forward(self, x):
        x = self.unpatch(x)
        batch, dur, seq, d = x.shape
        x = x.reshape(-1, seq, d)
        x = self.unpatchify(x)
        x = x.reshape(batch, dur, self.out_channels, self.height, self.width)
        return x
    
    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = self.height // p
        w = self.width // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = t.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs