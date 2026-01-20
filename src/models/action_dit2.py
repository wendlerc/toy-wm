from diffusers.utils import dummy_nvidia_modelopt_objects
from huggingface_hub.hub_mixin import DEFAULT_MODEL_CARD
import torch as t
from torch import nn
import torch.nn.functional as F
from jaxtyping import Float, Bool, Int
from torch import Tensor
from typing import Optional, Literal

from .dit2 import CausalDit
from vector_quantize_pytorch import VectorQuantize


class Codebook(nn.Module):
  # currently deviates from VectorQuantize module because it does not include down and up projections.
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

        self.learnt_actions1 = VectorQuantize(
            dim = d_model,
            codebook_size = n_actions,     # codebook size
            codebook_dim = d_actions,
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = beta,  # the weight on the commitment loss
            #use_cosine_sim = True,
            threshold_ema_dead_code = 2,
        )

        self.learnt_actions2 = VectorQuantize(
            dim = d_model,
            codebook_size = n_actions,     # codebook size
            codebook_dim = d_actions,
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = beta,  # the weight on the commitment loss
            #use_cosine_sim = True,
            threshold_ema_dead_code = 2,
        )
        self.action_dropout = action_dropout
        self.unconditional_action = nn.Parameter(t.randn(2, d_model)*d_model**(-0.5)) # todo the naming of vars is confusing right now. d_action should be actually the thing that is 16. 
        self.first = True

    @t._dynamo.disable
    def _quantize_actions1(self, actions_flat):
        # VectorQuantize uses view-heavy ops that can break Inductor when threshold_ema_dead_code is set.
        return self.learnt_actions1(actions_flat)
    
    @t._dynamo.disable
    def _quantize_actions2(self, actions_flat):
        # VectorQuantize uses view-heavy ops that can break Inductor when threshold_ema_dead_code is set.
        return self.learnt_actions2(actions_flat)

    def apply_action_dropout(self, actions_disc, labels, uncond):
        # during training we should apply action dropout replacing them with the unconditional action        
        # We drop actions with probability self.action_dropout
        dropout_mask = t.rand(actions_disc.shape[:2], device=actions_disc.device) < self.action_dropout
        # actions_disc shape: (b, dur, d)
        # Replace dropped actions with unconditional action embedding (broadcast correctly)
        uncond = uncond.view(1, 1, -1)
        # Use algebra to mask entries instead of clone & assignment
        actions_disc = actions_disc * (~dropout_mask).unsqueeze(-1) + uncond * dropout_mask.unsqueeze(-1)
        # Set dropped label indices to 0 for unconditional actions
        labels = labels * (~dropout_mask)  # dropped ones go to 0
        return actions_disc, labels

    def forward(self, z: Float[Tensor, "batch dur channels height width"]):
        actions_dummy = t.zeros(z.shape[0], z.shape[1], 2, device=z.device, dtype=t.int64)
        ts_dummy = t.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        # because for now we are reusing the action conditioned, fow-matching CausalDit we need to feed dummy actions and timesteps here...
        # TODO: create better abstractions, either by seperating DiT from rf-DiT or by reimplementing it here
        _, registers, _, _ = self.dit(z, actions_dummy, ts_dummy) 
        # batch x dur x 1 x d
        actions_cont = registers
        b, dur, _, d = actions_cont.shape
        actions_flat = actions_cont.reshape(-1, d)
        actions_disc1, labels1, loss1 = self._quantize_actions1(actions_flat)
        actions_disc2, labels2, loss2 = self._quantize_actions2(actions_flat)
        actions_disc1 = actions_disc1.reshape(b, dur, d)
        actions_disc2 = actions_disc2.reshape(b, dur, d)
        # Offset all labels by 1
        labels1 = labels1.reshape(b, dur) + 1
        labels2 = labels2.reshape(b, dur) + 1
        if self.training and self.action_dropout > 0.0:
            actions_disc1, labels1 = self.apply_action_dropout(actions_disc1, labels1, self.unconditional_action[0])
            actions_disc2, labels2 = self.apply_action_dropout(actions_disc2, labels2, self.unconditional_action[1])
        labels = t.cat([labels1[...,None], labels2[...,None]], dim=-1)
        loss = loss1 + loss2
        actions_disc = actions_disc1 + actions_disc2
        return actions_disc, actions_cont[:,:,0], labels, loss


def get_model(height, width, 
              n_window= 5, 
              d_model= 64, 
              action_dropout = 0.0,
              n_actions = 6,
              d_actions = 64,
              beta = 0.25,
              T=100, 
              n_blocks=2, 
              patch_size=2, 
              n_heads=8, 
              bidirectional=False, 
              in_channels=3, 
              C=10000, 
              rope_type: Literal["rope", "learn", "vid"] = "rope",
              use_flex=False):
    return ActionDit(height, width,
                     n_window, 
                     d_model, 
                     action_dropout=action_dropout,
                     n_actions=n_actions,
                     d_actions=d_actions,
                     beta=beta,
                     T=T, 
                     in_channels=in_channels, 
                     n_blocks=n_blocks, 
                     patch_size=patch_size, 
                     n_heads=n_heads, 
                     bidirectional=bidirectional, 
                     rope_C=C, 
                     rope_type=rope_type,
                     use_flex=use_flex)
        
