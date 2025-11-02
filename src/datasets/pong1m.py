from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch as t
import numpy as np

def get_loader(batch_size=64, fps=30, duration=5, shuffle=True):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy"))
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy"))
    height, width, channels = frames.shape[-3:]
    n = frames.shape[0]//(fps*duration) 
    frames = frames[:n*fps*duration]
    frames = frames.reshape(n, fps*duration, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    frames = frames.float()
    frames /= 255.0
    mean = frames.mean(dim=(0,1,3,4), keepdim=True)
    std = frames.std(dim=(0,1,3,4), keepdim=True)
    frames = (frames - mean) / (std + 1e-6)
    # frames = 2*frames - 1 # this creates nans
    actions = actions[:n*fps*duration]
    actions = actions.reshape(-1, fps*duration)
    dataset = TensorDataset(frames, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, mean, std