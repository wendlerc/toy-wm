import torch as t
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from pdb import set_trace

import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb

from torch.nn import functional as F
from tqdm import tqdm
import math
import random

from muon import SingleDeviceMuonWithAuxAdam

from ..inference.sampling import sample


def log_video(z, tag="generated_video", fps=5):
    """
    Log a video to wandb from a tensor of frames.
    z: (B, T, C, H, W) or (T, C, H, W) in [0,1] float
    """
    if z.ndim == 5:
        frames = z[0]        # take first in batch
    elif z.ndim == 4:
        frames = z
    else:
        raise ValueError(f"Unexpected shape: {z.shape}")
    #frames_uint8 = (((frames.clamp(-1, 1) + 1)/2) * 255).byte().cpu().numpy()
    #frames_uint8 = ((F.tanh(frames)+1)/2*255).byte().cpu().numpy()
    frames_uint8 = frames.byte().cpu().numpy()
    wandb.log({"sample": wandb.Video(frames_uint8, fps=fps, format="mp4")})


def train(model, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, max_steps=1000, 
          p_pretrain=1.0,
          clipping=True,
          checkpoint_manager=None):

    device = model.device
    dtype = model.dtype
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

    body_weights = list(model.blocks.parameters())
    other_weights = set(model.parameters()) - set(body_weights)

    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]
    nonhidden_params = list(other_weights)
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
            lr=lr1, weight_decay=weight_decay),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
            lr=lr2, betas=betas, weight_decay=weight_decay),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    # Use CosineAnnealingWarmRestarts with a warmup period by combining with a LambdaLR for linear warmup.
    # Here, we first do linear warmup for warmup_steps, then cosine annealing
    warmup_steps = 100
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # after warmup: cosine annealing from warmup_steps to max_steps
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        #set_trace()
        optimizer.zero_grad()
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, actions = next(iterator)
        
        actions += 1
        frames[:, 1:] = frames[:, :-1]
        frames[:, 0] = 0
        if random.random() < p_pretrain:
            actions[:, 2:] = actions[:, :-2] 
            actions[:, :2] = 0
            mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
            actions[mask] = 0
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
        else:
            actions[:,1:] = actions[:, :-1]
            actions[:,0] = 0
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            ts[:, :-1] = 1 - (ts[:, :-1]*0.5 + 0.5)
        frames = frames[:, :model.n_window]
        actions = actions[:, :model.n_window]
        frames = frames.to(device).to(dtype)
        actions = actions.to(device)
        ts = ts[:,:model.n_window]
        z = t.randn_like(frames, device=device, dtype=dtype)
        x0 = frames
        vel_true = x0 - z
        x_t = x0 - ts[:, :, None, None, None].to(device) * vel_true
        vel_pred = model(x_t, actions, ts)
        loss = F.mse_loss(vel_pred, vel_true, reduction="mean")
        wandb.log({"loss": loss.item()})
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())

        if step % 100 == 0 and pred2frame is not None:
            checkpoint_manager.save(metric=loss.item(), step=step, model=model, optimizer=optimizer, scheduler=scheduler)
            # compute loss per noise level
            noise_levels = [1., 0.75, 0.5, 0.25, 0.1, 0]
            noise_losses = []
            with t.no_grad():
                for noise_level in noise_levels:
                    z = t.randn_like(frames, device=device, dtype=dtype)
                    x0 = frames
                    vel_true = x0 - z
                    ts = noise_level * t.ones(frames.shape[0], frames.shape[1], device=device, dtype=dtype)
                    x_t = x0 - ts[:, :, None, None, None].to(device) * vel_true
                    vel_pred = model(x_t, actions, ts)
                    noise_losses.append(F.mse_loss(vel_pred, vel_true, reduction="mean"))
                    wandb.log({f"noise:{noise_level}": noise_losses[-1].item()})


            if frames.shape[1] == 1: 
                z_sampled = sample(model, 
                                   t.randn_like(frames[:30], device=device, dtype=dtype), 
                                   actions[:30], num_steps=10)
                z_sampled = z_sampled.permute(1, 0, 2, 3, 4)
            else:
                z_sampled = sample(model, t.randn_like(frames[:1], device=device, dtype=dtype), actions[:1], num_steps=10)
            frames_sampled = pred2frame(z_sampled)
            log_video(frames_sampled, tag=f"{step:04d}")

    return model