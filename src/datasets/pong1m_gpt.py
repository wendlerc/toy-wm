import torch as t
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def pred2frame(y, lam=1e-6):
    #x01 = (t.sigmoid(y) - lam)/(1 - 2*lam)
    #frames = (x01.clamp(0,1) * 255.0).round().to(t.uint8)
    y = y.clamp(0, 1)
    frames = (y * 255.0).round().to(t.uint8)
    return frames


# --------- DATA LOADER ---------
def get_loader(batch_size=64, fps=30, duration=5, shuffle=True, debug=False):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy")).to(t.uint8)
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy")).long()

    height, width, channels = frames.shape[-3:]
    n = frames.shape[0] // (fps * duration)

    frames = frames[:n*fps*duration]
    frames = frames.reshape(n, fps*duration, height, width, channels)
    frames = frames.permute(0, 1, 4, 2, 3)
    actions = actions[:n*fps*duration]
    actions = actions.reshape(-1, fps*duration)

    x01 = frames.float()/255.0
    u = t.rand_like(x01)/256.0           # uniform dequantization
    x01 = (x01 + u).clamp(0, 1)
    lam = 1e-6
    frames = t.log(t.clamp(lam + (1-2*lam)*x01, min=1e-12)) \
        - t.log(1 - t.clamp(lam + (1-2*lam)*x01, max=1-1e-12))
   
    if debug:
        print("🔍 DEBUG MODE: freezing video to 1 sample & 1 frame")
        frames = frames[:1, :1]                   # (1,1,H,W,3)
        actions = actions[:1, :1]                 # (1,1)

    dataset = TensorDataset(frames, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"✅ Loaded {len(dataset)} episodes → {len(loader)} batches")
    return loader, pred2frame
