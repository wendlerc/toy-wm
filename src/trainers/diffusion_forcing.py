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

from muon import SingleDeviceMuonWithAuxAdam


mean = t.tensor([[[[[0.0352]],
                    [[0.1046]],
                    [[0.1046]]]]]) 
std = t.tensor([[[[[0.1066]],
                    [[0.0995]],
                    [[0.0995]]]]])


@t.no_grad()
def sample(v, z, frames, actions, num_steps=10):
    device = v.device
    ts = 1 - t.linspace(0, 1, num_steps+1, device=device)
    ts = 3*ts/(2*ts + 1)
    z_prev = z.clone()
    z_prev = z_prev.to(device)
    for i in tqdm(range(len(ts)-1)):
        t_cond = ts[i].repeat(z_prev.shape[0], 1)
        z_prev = z_prev + (ts[i] - ts[i+1])*v(z_prev.to(device), actions.to(device), t_cond.to(device)) 
    return z_prev


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

    # Convert to uint8 for video logging (T, C, H, W)
    # boxplot frames
    plt.boxplot(frames.flatten().cpu())
    plt.savefig(f"{tag}_boxplot.png")
    plt.close()
    frames_uint8 = (frames.clamp(0, 1) * 255).byte().cpu().numpy()
    print(frames_uint8.shape)

    wandb.log({
        tag: wandb.Video(frames_uint8, fps=fps, format="mp4")
    })


def train(model, dataloader, lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, max_steps=1000):

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
        frames = frames.to(device).to(dtype)
        actions = actions.to(device)
        actions[:, 1:] = actions[:, :-1] + 1
        actions[:, 0] = 0

        z = t.randn_like(frames, device=device, dtype=dtype)
        x0 = frames
        vel_true = x0 - z
        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
        x_t = x0 - ts[:, :, None, None, None].to(device) * vel_true
        vel_pred = model(x_t, actions, ts)
        loss = F.mse_loss(vel_pred, vel_true, reduction="mean")
        wandb.log({"loss": loss.item()})
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        if step % 100 == 0:
            z_sampled = sample(model, t.randn_like(frames[:1], device=device, dtype=dtype), frames[:1],actions[:1], num_steps=10)
            z_sampled = z_sampled.cpu()*std + mean
            log_video(z_sampled, tag=f"{step:04d}")

    return model