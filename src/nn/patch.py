from torch import nn

class Patch(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=3)

    def forward(self, x):
        assert x.shape[-1] % self.patch_size == 0 and x.shape[-2] % self.patch_size == 0, f"shape not compatible with patching, both dims need to be divisible by patch_size {self.patch_size}"
        # input: (batch, time, channels, height, width)
        batch, dur, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        x = self.conv(x) 
        x = x.reshape(batch, dur, self.out_channels, h//self.patch_size, w//self.patch_size)
        # output: (batch, time, channels, height/patch_size, width/patch_size)
        x = self.flatten(x)
        # output: (batch, time, channels, (height/patch_size) * (width/patch_size))
        x = x.permute(0, 1, 3, 2)
        # output: (batch, time, (height/patch_size) * (width/patch_size), channels)
        return x

class UnPatch(nn.Module):
    def __init__(self, height, width, in_channels=64, out_channels=3, patch_size=2):
        super().__init__()
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unpatch = nn.Linear(64, out_channels*patch_size**2)

    def forward(self, x):
        x = self.unpatch(x)
        batch, dur, seq, d = x.shape
        x = x.reshape(batch, dur, seq*self.patch_size**2, self.out_channels)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch, dur, self.out_channels, self.height, self.width)
        return x