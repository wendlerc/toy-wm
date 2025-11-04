from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch as t
import numpy as np
from einops import rearrange

mean = t.tensor([[[[[0.0352]],
                    [[0.1046]],
                    [[0.1046]]]]]) 
std = t.tensor([[[[[0.1066]],
                    [[0.0995]],
                    [[0.0995]]]]])

def fixed2frame(y, lam=1e-6):
    y = y.clamp(-1, 1) * 0.5 + 0.5
    frames = (y * 255.0).round().byte()
    return frames

def z2frame(y, lam=1e-6, mean=mean, std=std):
    y = y*std.to(y.dtype).to(y.device) + mean.to(y.dtype).to(y.device)
    frames = (y.clamp(0, 1) * 255.0).round().byte()
    return frames

def get_loader(batch_size=64, fps=30, duration=5, shuffle=True, debug=False, mode="z", mean=mean, std=std):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy"))
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy"))
    height, width, channels = frames.shape[-3:]
    n = frames.shape[0]//(fps*duration) 
    frames = frames[:n*fps*duration]
    frames = frames.reshape(n, fps*duration, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    actions = actions[:n*fps*duration]
    actions = actions.reshape(-1, fps*duration)
    b, dur, c, h, w = frames.shape
    if mode == "-1,1":
        z = rearrange(frames, "b dur c h w -> (b dur h w) c")
        mask = (z == t.tensor([6, 24, 24], dtype=z.dtype)).all(dim=1)
        z = (z.float()/255.0 - 0.5)*2
        z[mask] = 0
        z = rearrange(z, "(b dur h w) c -> b dur c h w", b=b, dur=dur, c=c, h=h, w=w)
        frames = z
        pred2frame = fixed2frame
    elif mode == "z":
        frames = frames.float()/255.0
        frames = (frames - mean) / (std + 1e-6)
        pred2frame = z2frame
    else:
        raise ValueError(f"Invalid mode: {mode}")

    firstf = frames[0]
    firsta = actions[0]
    if debug:
        frames = 0*frames + firstf[None]
        actions = 0*actions + firsta[None]
        frames = 0*frames + frames[:,0].unsqueeze(1)
    dataset = TensorDataset(frames, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"{frames.shape[0]//batch_size} batches")
    return loader, pred2frame