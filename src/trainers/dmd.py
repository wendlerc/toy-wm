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

from ..utils import log_video, load_model_from_config

def get_muon(model, lr1, lr2, betas, weight_decay):
    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]

    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]
    nonhidden_params = list(other_weights)

    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=lr1,
            weight_decay=weight_decay,
        ),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=lr2,
            betas=betas,
            weight_decay=weight_decay,
        ),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    return optimizer


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
          n_fake_updates=3, 
          device=None, dtype=None):


    true_v = load_model_from_config(cfg)
    fake_v = load_model_from_config(cfg)
    gen = load_model_from_config(cfg)

    true_v.to(device).to(dtype)
    fake_v.to(device).to(dtype)
    gen.to(device).to(dtype)

    for p in true_v.parameters():
        p.requires_grad = False

    true_v = t.compile(true_v)
    fake_v = t.compile(fake_v)
    gen = t.compile(gen)
    
    device = gen.device
    dtype = gen.dtype
    print(device, dtype)
    
    fake_opt = get_muon(fake_v, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))
    gen_opt = get_muon(gen, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))

    fake_sched = t.optim.lr_scheduler.LambdaLR(fake_opt, partial(lr_lambda, max_steps=n_fake_updates*max_steps))
    gen_sched = t.optim.lr_scheduler.LambdaLR(gen_opt, partial(lr_lambda, max_steps=max_steps))
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
        frames = frames.to(device)
        actions[:, 1:] = actions[:, :-1] 
        actions[:, :1] = 0
        mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
        actions[mask] = 0
        
        # generate a video
        # Vectorized mask creation: gen_ts is 1 where frame >= frame_id, 0 otherwise
        frame_indices = t.arange(frames.shape[1], device=device)[None, :]  # (1, T)
        batch_indices = t.arange(frames.shape[0], device=device)
        
        frame_ids = t.randint(0, frames.shape[1], (frames.shape[0],), device=device, dtype=t.int32)
        gen_ts = (frame_indices >= frame_ids[:, None]).to(dtype)  # (B, T)

        actions = actions.to(device)
        z = t.randn_like(frames, device=device, dtype=dtype)
        x_inp = frames - gen_ts[:,:,None,None,None]*(frames - z)
        v_pred = gen(x_inp, actions, gen_ts)
        x_pred = z + v_pred
        
        # compute dmd gradient
        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
        x_t = x_pred - ts[:,:,None,None,None]*v_pred 
        x_t_nograd = x_t.detach()
        real_vel = true_v(x_t_nograd, actions, ts)
        fake_vel = fake_v(x_t_nograd, actions, ts)

        gen_loss = 0.5*F.mse_loss(x_t[batch_indices, frame_ids], x_t_nograd[batch_indices, frame_ids] + (real_vel[batch_indices, frame_ids].detach() - fake_vel[batch_indices, frame_ids].detach()))
        gen_loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        gen_opt.step()
        gen_sched.step()
        wandb.log({"gen_loss": gen_loss.item()})
        wandb.log({"gen_lr": gen_sched.get_last_lr()[0]})
        
        # update fake_v
        fake_loss = F.mse_loss(fake_vel[batch_indices, frame_ids], v_pred[batch_indices, frame_ids].detach())
        fake_loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(fake_v.parameters(), 1.0)
        fake_opt.step()
        fake_sched.step()
        wandb.log({"fake_loss": fake_loss.item()})
        wandb.log({"fake_lr": fake_sched.get_last_lr()[0]})
        for _ in range(n_fake_updates-1): #TODO this needs to be fixed to be the same as above
            fake_opt.zero_grad()
            frames, actions = next(iterator)
            actions += 1
            frames[:, 1:] = frames[:, :-1]
            frames[:, 0] = 0
            frames = frames.to(device)
            actions[:, 1:] = actions[:, :-1] 
            actions[:, :1] = 0
            mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
            actions[mask] = 0
            actions = actions.to(device)

            batch_indices = t.arange(frames.shape[0], device=device)
            frame_indices = t.arange(frames.shape[1], device=device)[None, :]  # (1, T)
            frame_ids = t.randint(0, frames.shape[1], (frames.shape[0],), device=device, dtype=t.int32)
            gen_ts = (frame_indices >= frame_ids[:, None]).to(dtype)  # (B, T)

            z = t.randn_like(frames, device=device, dtype=dtype)
            x_inp = frames - gen_ts[:,:,None,None,None]*(frames - z)
            v_pred = gen(x_inp, actions, gen_ts)
            x_pred = z + v_pred
            
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            x_t = x_pred - ts[:,:,None,None,None]*v_pred 
            x_t_nograd = x_t.detach()
            fake_vel = fake_v(x_t_nograd, actions, ts)

            fake_loss = F.mse_loss(fake_vel[batch_indices, frame_ids], v_pred[batch_indices, frame_ids].detach())
            fake_loss.backward()
            if clipping:
                t.nn.utils.clip_grad_norm_(fake_v.parameters(), 1.0)
            fake_opt.step()
            fake_sched.step()
            wandb.log({"fake_loss": fake_loss.item()})
            wandb.log({"fake_lr": fake_sched.get_last_lr()[0]})

        pbar.set_postfix_str(f'loss_gen {gen_loss.item():.4f} loss_fake {fake_loss.item():.4f}')

        if step % 100 == 0 and pred2frame is not None:
            with t.no_grad():
                eval_loss = F.mse_loss(x_pred[batch_indices, frame_ids], frames[batch_indices, frame_ids])
                wandb.log({"eval_loss":eval_loss})
                checkpoint_manager.save(metric=eval_loss.item(), step=step, model=gen, optimizer=gen_opt, scheduler=None)
                frame_preds = x_pred[batch_indices, frame_ids]
                frames_sampled = pred2frame(frame_preds.detach().cpu())
                log_video(frames_sampled)
    return gen