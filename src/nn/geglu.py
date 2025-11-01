from torch import nn

class GEGLU(nn.Module):
    def __init__(self, d_in, d_mid, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_mid = d_mid
        self.d_out = d_out
        self.up_proj = nn.Linear(d_in, d_mid)
        self.up_gate = nn.Linear(d_in, d_mid)
        self.down = nn.Linear(d_mid, d_out)
        self.nonlin = nn.GELU()
    
    def forward(self, x):
        x = self.up_proj(x) * self.nonlin(self.up_gate(x))
        x = self.down(x)
        return x