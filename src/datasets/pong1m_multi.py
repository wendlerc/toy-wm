from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch as t
import numpy as np
from einops import rearrange

def fixed2frame(y, lam=1e-6):
    y = y.clamp(-1, 1) * 0.5 + 0.5
    frames = (y * 255.0).round().byte()
    return frames

def prep(frames, actions, n, frames_per_example):
    height, width, channels = frames.shape[-3:]
    frames = frames[:n*frames_per_example]
    frames = frames.reshape(n, frames_per_example, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    actions = actions[:n*frames_per_example]
    actions = actions.reshape(-1, frames_per_example)
    return frames, actions

def grey_background(frames):
    b, dur, c, h, w = frames.shape
    z = rearrange(frames, "b dur c h w -> (b dur h w) c")
    mask = (z == t.tensor([6, 24, 24], dtype=z.dtype)).all(dim=1)
    z[mask] = 255//2
    z = rearrange(z, "(b dur h w) c -> b dur c h w", b=b, dur=dur, c=c, h=h, w=w)
    return z

def switch_player_colors(frames):
    b, dur, c, h, w = frames.shape
    z = rearrange(frames, "b dur c h w -> (b dur h w) c")
    cyan = t.tensor([0, 255, 255], dtype=z.dtype)
    red = t.tensor([255, 0, 0], dtype=z.dtype)
    mask_cyan = (z == cyan).all(dim=1)
    mask_red = (z == red).all(dim=1)
    print(mask_cyan.sum(), mask_red.sum())
    z[mask_cyan] = z[mask_cyan] * 0 + red.unsqueeze(0)
    z[mask_red] = z[mask_red]* 0 + cyan.unsqueeze(0)
    z = rearrange(z, "(b dur h w) c -> b dur c h w", b=b, dur=dur, c=c, h=h, w=w)
    return z


def get_loader(batch_size=64, fps=30, duration=5, shuffle=True, pin_memory=True, num_workers=4):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy"))
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy"))
    frames_mirrored = t.from_numpy(np.load("./datasets/pong1M/frames_mirrored.npy"))
    actions_mirrored = t.from_numpy(np.load("./datasets/pong1M/actions.npy"))
    frames_per_example = fps*duration + 1
    n = frames.shape[0]//frames_per_example
    frames, actions = prep(frames, actions, n, frames_per_example)
    frames_mirrored, actions_mirrored = prep(frames_mirrored, actions_mirrored, n, frames_per_example)
    frames = grey_background(frames)
    frames_mirrored = grey_background(frames_mirrored)
    frames_mirrored = switch_player_colors(frames_mirrored)
    actions += 1
    actions_mirrored += 1
    actions_uncond = t.zeros_like(actions, dtype=actions.dtype)

    actions_cyan = t.cat([actions, actions_uncond], dim=0)
    actions_red = t.cat([actions_uncond, actions_mirrored], dim=0)
    frames_both = t.cat([frames, frames_mirrored], dim=0)
    frames_both = (frames_both.float()/255.0 - 0.5)*2

    frames_both = frames_both[:, 1:]
    actions_cyan = actions_cyan[:, :-1]
    actions_red = actions_red[:, :-1] 
    
    dataset = TensorDataset(frames_both, actions_cyan, actions_red)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    print(f"{frames.shape[0]//batch_size} batches")
    return loader, fixed2frame