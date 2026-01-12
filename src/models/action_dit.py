import torch as t
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional, Literal

from .dit import CausalDit

class Codebook(nn.Module):
  def __init__(self, n, d, beta=0.25, dropout_p=0.0):
    super().__init__()
    init = t.randn(n, d)
    self.beta = beta
    self.D = nn.Parameter(init)
    self.dropout_p = dropout_p

  def forward(self, x):
    # x ... batch x d
    dots = x @ self.D.T # batch x n
    dots = F.dropout(dots, p=self.dropout_p, training=self.training)
    dists = (x**2).sum(dim=1, keepdims=True) - 2*dots + (self.D.T**2).sum(dim=0, keepdims=True)
    argmins = t.argmin(dists, dim=1)
    x_q = self.D[argmins]
    x_st = x + (x_q - x).detach()
    codebook_loss = F.mse_loss(x_q, x.detach())
    commit_loss = F.mse_loss(x, x_q.detach())
    loss = codebook_loss + self.beta * commit_loss
    return x_st, argmins, loss

class ActionDit(nn.Module):
    def __init__(self, height, width, n_window, d_model, 
                       d_actions, beta=0.25, action_dropout = 0.0,
                       T=1000, in_channels=3,
                       patch_size=2, n_heads=8, expansion=4, n_blocks=6, n_actions=4, bidirectional=False, 
                       debug=False, 
                       rope_C=10000,
                       rope_tmax=None,
                       rope_type: Literal["rope", "learn", "vid"] = "rope",
                       use_flex: bool = False):
        super().__init__()
        self.dit = CausalDit(height, width, n_window, d_model, T=T, in_channels=in_channels,
                             patch_size=patch_size, n_heads=n_heads, expansion=expansion, n_blocks=n_blocks,
                             n_registers=1, n_actions=1, bidirectional=bidirectional,
                             debug=debug, rope_C=rope_C, rope_tmax=rope_tmax, rope_type=rope_type, use_flex=use_flex,
                             return_registers=True)

        self.action_head = nn.Linear(d_model, d_actions)
        self.learnt_actions = Codebook(n_actions, d_actions, beta=beta, dropout_p=action_dropout)

    def forward(self, z: Float[Tensor, "batch dur channels height width"]):
        actions = t.zeros(z.shape[0], z.shape[1], device=z.device, dtype=t.int64)
        ts = t.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        # because for now we are reusing the action conditioned, fow-matching CausalDit we need to feed dummy actions and timesteps here...
        # TODO: create better abstractions, either by seperating DiT from rf-DiT or by reimplementing it here
        _, registers, _, _ = self.dit(z, actions, ts) 
        # batch x dur x 1 x d
        actions_cont = self.action_head(registers)
        b, dur, _, d = actions_cont.shape
        actions_flat = actions_cont.reshape(-1, d)
        actions_disc, labels, loss = self.learnt_actions(actions_flat)
        actions_dict = actions_disc.reshape(b, dur, d)
        return actions_dict, labels, loss


def get_model(height, width, 
              n_window= 5, 
              d_model= 64, 
              n_actions = 6,
              d_actions = 64,
              T=100, 
              n_blocks=2, 
              patch_size=2, 
              n_heads=8, 
              bidirectional=False, 
              in_channels=3, 
              C=10000, 
              rope_type: Literal["rope", "learn", "vid"] = "rope",
              use_flex=False):
    assert d_model == d_actions, f"Currently, we only support {d_model} == {d_actions}."
    return ActionDit(height, width,
                     n_window, 
                     d_model, 
                     n_actions=n_actions,
                     d_actions=d_actions,
                     T=T, 
                     in_channels=in_channels, 
                     n_blocks=n_blocks, 
                     patch_size=patch_size, 
                     n_heads=n_heads, 
                     bidirectional=bidirectional, 
                     rope_C=C, 
                     rope_type=rope_type,
                     use_flex=use_flex)
        
