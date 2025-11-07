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
from functools import partial

from muon import SingleDeviceMuonWithAuxAdam

from ..inference import sample
from ..utils import log_video
from ..utils

def get_muon(model):
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
    return optimizier

def lr_lambda(current_step, max_steps, warmup_steps=100):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train(cfg, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, max_steps=1000, 
          p_pretrain=1.0,
          clipping=True,
          checkpoint_manager=None,
          n_fake_updates=5):


    true_v = load_model_from_config(cfg)
    fake_v = load_model_from_config(cfg)
    gen = load_model_from_config(cfg)

    for p in true_v.parameters():
        p.requires_grad = False
    
    device = gen.device
    dtype = gen.dtype
    print(device, dtype)
    
    fake_opt = get_muon(fake_v)
    gen_opt = get_muon(generator)

    fake_sched = t.optim.lr_scheduler.LambdaLR(fake_opt, partial(lr_lambda, max_steps=n_fake_updates*max_steps))
    gen_sched = t.optim.lr_scheduler.LambdaLR(gen_opt, partial(lr_lamda, max_steps=max_steps))
    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        #set_trace()
        fake_opt.zero_grad()
        gen_opt.zero_grad()
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, actions = next(iterator)
        # prep the data
        actions += 1
        frames[:, 1:] = frames[:, :-1]
        frames[:, 0] = 0

        actions[:, 1:] = actions[:, :-1] 
        actions[:, :1] = 0
        mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
        actions[mask] = 0
        
        # generate a video
        gen_ts = t.ones_like(ts, device=device, dtype=dtype)
        actions = actions.to(device)
        z = t.randn_like(frames, device=device, dtype=dtype)
        v_pred = gen(z, actions, gen_ts)
        x_pred = z + v_pred
        
        # compute dmd gradient
        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
        x_t = x_pred - ts*v_pred # maybe use fresh noise here?
        x_t_nograd = x_t.nograd()
        real_vel = real_v(x_t_nograd, actions, ts)
        fake_vel = fake_v(x_t_nograd, actions, ts)

        gen_loss = 0.5*F.mse_loss(x_t, x_t_nograd - (fake_vel.detach() - real_vel.detach()))
        gen_loss.backward()
        gen_opt.step()
        gen_sched.step()
        wandb.log("gen_loss", gen_loss.item())
        wandb.log("gen_lr", gen_sched.get_last_lr())
        
        # update fake_v
        fake_loss = F.mse_loss(fake_vel, v_pred.detach())
        fake_loss.backward()
        fake_opt.step()
        wandb.log("fake_loss", fake_loss.item())
        wandb.log("fake_lr", fake_sched.get_last_lr())
        for _ in range(n_fake_updates-1):
            fake_opt.zero_grad()
            v_pred = gen(z, actions, gen_ts)
            x_pred = z + v_pred            
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            x_t = x_pred - ts*v_pred
            fake_vel = fake_v(x_t_nograd, actions, ts)
            fake_loss = F.mse_loss(fake_vel, v_pred.detach())
            fake_loss.backward()
            fake_opt.step()
            wandb.log("fake_loss", fake_loss.item())
            wandb.log("fake_lr", fake_sched.get_last_lr())

        pbar.set_postfix_str(f'loss_gen {l_gen.item():.4f} loss_fake {l_fake.item():.4f}')

        if step % 100 == 0 and pred2frame is not None:
            checkpoint_manager.save(metric=loss.item(), step=step, model=gen_v, optimizer=gen_opt, scheduler=gen_sched)
            frames_sampled = pred2frame(x_pred.detach().cpu())
            log_video(frames_sampled)
    return model