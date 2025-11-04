from torch import nn

class GEGLU(nn.Module):
    def __init__(self, d_in, d_mid, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_mid = d_mid
        self.d_out = d_out
        self.up_proj = nn.Linear(d_in, d_mid, bias=True)
        self.up_proj.bias.data.zero_()
        self.up_gate = nn.Linear(d_in, d_mid, bias=True)
        self.up_gate.bias.data.zero_()
        self.down = nn.Linear(d_mid, d_out, bias=True)
        self.down.bias.data.zero_()
        self.nonlin = nn.SiLU()
    
    def forward(self, x):
        x = self.up_proj(x) * self.nonlin(self.up_gate(x))
        x = self.down(x)
        return x