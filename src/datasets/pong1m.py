from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch as t
import numpy as np
from einops import rearrange

def fixed2frame(y, lam=1e-6):
    y = y.clamp(-1, 1) * 0.5 + 0.5
    frames = (y * 255.0).round().byte()
    return frames

def get_loader(batch_size=64, fps=30, duration=5, shuffle=True, debug=False, drop_duration=False):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy"))
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy"))
    height, width, channels = frames.shape[-3:]
    frames_per_example = fps*duration + 1
    n = frames.shape[0]//frames_per_example
    frames = frames[:n*frames_per_example]
    frames = frames.reshape(n, frames_per_example, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    actions = actions[:n*frames_per_example]
    actions = actions.reshape(-1, frames_per_example)
    b, dur, c, h, w = frames.shape
    # preprocess
    z = rearrange(frames, "b dur c h w -> (b dur h w) c")
    mask = (z == t.tensor([6, 24, 24], dtype=z.dtype)).all(dim=1)
    z = (z.float()/255.0 - 0.5)*2
    z[mask] = 0
    z = rearrange(z, "(b dur h w) c -> b dur c h w", b=b, dur=dur, c=c, h=h, w=w)
    frames = z
    pred2frame = fixed2frame

    # we add 1 so 0 becomes our dummy action for CFG
    actions += 1
    # each frame gets as a condition the action of its previous frame
    frames = frames[:, 1:]
    actions = actions[:, :-1] 

    # 0... dummy
    # 1... do nothing
    # 2... up
    # 3... down
    firstf = frames[0]
    firsta = actions[0]
    if debug:
        frames = 0*frames + firstf[None]
        actions = 0*actions + firsta[None]
        frames = 0*frames + frames[:,0].unsqueeze(1)
    if drop_duration:
        dataset = TensorDataset(frames[:, 0], actions[:,0]*0)
    else:
        dataset = TensorDataset(frames, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"{frames.shape[0]//batch_size} batches")
    return loader, pred2frame