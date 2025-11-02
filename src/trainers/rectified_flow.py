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

mean = t.tensor([[[[[0.0352]],
                    [[0.1046]],
                    [[0.1046]]]]]) 
std = t.tensor([[[[[0.1066]],
                    [[0.0995]],
                    [[0.0995]]]]])


@t.no_grad()
def sample(v, z, frames, actions, num_steps=50, uniform=False):
    device = v.device
    if uniform: 
        ts = 1 - t.linspace(0, 1, num_steps+1, device=device)
    else:
        ts = 1 - F.sigmoid(t.randn(num_steps+1, device=device).msort())
    z_prev = z.clone()
    z_prev = z_prev.to(device)
    for i in tqdm(range(len(ts)-1)):
        t_cond = ts[i].repeat(z_prev.shape[0], 1)
        z_prev = z_prev + (ts[i] - ts[i+1])*v(z_prev.to(device), frames[:, :-1].to(device), actions.to(device), t_cond.to(device)) 
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


def train(model, dataloader, lr=1e-2, weight_decay=1e-5, max_steps=1000):

    device = model.device
    dtype = model.dtype
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        #set_trace()
        optimizer.zero_grad()
        frames, actions = next(iterator)
        frames = frames.to(device).to(dtype)
        actions = actions.to(device)

        z = t.randn_like(frames, device=device, dtype=dtype)
        x0 = frames
        vel_true = x0 - z
        ts = F.sigmoid(t.randn(frames.shape[0], 1, device=device, dtype=dtype))
        x_t = x0 - ts[:, :, None, None, None].to(device) * vel_true
        vel_pred = model(x_t, x0[:,:-1], actions, ts)
        loss = F.mse_loss(vel_pred, vel_true)
        wandb.log({"loss": loss.item()})
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        if step % 100 == 0:
            z_sampled = sample(model, t.randn_like(frames[:1], device=device, dtype=dtype), frames[:1], actions[:1])
            z_sampled = z_sampled.cpu()*std + mean
            log_video(z_sampled, tag=f"generated_gif_{step}")

    return model