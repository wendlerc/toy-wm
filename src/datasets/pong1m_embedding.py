import torch as t
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# --------- CLASS EMBEDDINGS ---------
CLASSES = t.tensor([
    [ 1.26627204, -1.94306927, -2.00183854, -0.01164514,  1.30366786, -2.15118120],
    [-0.88698881, -0.89533847,  1.41444045, 0.12760728,  0.54124583, -1.27942701],
    [-0.57892137,  2.03835674, -0.66793195, -2.30802651, -0.20119850,  2.14320196],
    [-0.68151154,  0.53591260,  1.20395372, -0.20675289,  0.96128205, -0.00090202],
    [ 0.15378938,  1.64493096, -1.14575637, -0.38769038,  0.34031443, -0.74085295]], dtype=t.float32)

color2class = {
    (0,   0,   0): CLASSES[0],   # black
    (0, 255, 255): CLASSES[1],   # cyan
    (6,  24,  24): CLASSES[2],   # dark teal
    (255, 0,   0): CLASSES[3],   # red
    (255,255,255): CLASSES[4],   # white
}

# Build lookup tables
KEYS = t.tensor(list(color2class.keys()), dtype=t.uint8)   # (5, 3)
VALS = CLASSES[:len(KEYS)]                                  # (5, 3)


# --------- ENCODER: RGB → CLASS EMBEDDING ---------
def pred2frame(frames, keys=KEYS, vals=VALS):
    """
    frames: (N, T, H, W, 3) uint8
    returns: (N, T, H, W, D) float32
    """
    frames_flat = frames.reshape(-1, 3)                          # (pixels, 3)
    eq = (frames_flat[:, None, :] == keys[None, :, :]).all(-1)   # (pixels, K)

    if not eq.any(-1).all():
        bad = frames_flat[~eq.any(-1)]
        raise ValueError(f"Found unknown pixel colors: {bad[:10]} ...")

    # ✅ MPS-safe: cast bool → int before argmax
    idx = eq.int().argmax(dim=1)                                 # (pixels,)
    return vals[idx].reshape(*frames.shape[:-1], -1)             # (N,T,H,W,D)


# --------- DECODER: CLASS EMBEDDING → RGB ---------
RGB_KEYS = KEYS.clone().to(t.uint8)
EMB_VALS = VALS.clone().to(t.float32)

def decode_class_to_rgb(emb, keys=RGB_KEYS, vals=EMB_VALS):
    """
    emb:  (N, T, H, W, D) float32
    returns: (N, T, H, W, 3) uint8
    """
    keys = keys.to(emb.device)
    vals = vals.to(emb.device)
    emb = emb.permute(0, 1, 3, 4, 2)
    emb_flat = emb.reshape(-1, emb.shape[-1])                  # (pixels, D)

    # L2 distance to each class vector
    dists = ((emb_flat[:, None, :] - vals[None, :, :]) ** 2).sum(-1)  # (pixels, K)
    idx = dists.argmin(dim=1)                                         # (pixels,)
    rgb = keys[idx]
    return rgb.reshape(*emb.shape[:-1], 3).permute(0, 1, 4, 2, 3)                           # restore shape


# --------- DATA LOADER ---------
def get_loader(batch_size=64, fps=30, duration=5, shuffle=True, debug=False):
    frames = t.from_numpy(np.load("./datasets/pong1M/frames.npy")).to(t.uint8)
    actions = t.from_numpy(np.load("./datasets/pong1M/actions.npy")).long()

    height, width, channels = frames.shape[-3:]
    n = frames.shape[0] // (fps * duration)

    frames = frames[:n * fps * duration]
    actions = actions[:n * fps * duration]

    frames = frames.reshape(n, fps * duration, height, width, channels)
    actions = actions.reshape(n, fps * duration)

    # Encode RGB → embedding
    frames = map_rgb_to_class(frames).float()     # now (n,T,H,W,3) float32
    frames = frames.permute(0, 1, 4, 2, 3)        # (n,T,3,H,W)
    if debug:
        print("🔍 DEBUG MODE: freezing video to 1 sample & 1 frame")
        frames = frames[:1, :1]                   # (1,1,H,W,3)
        actions = actions[:1, :1]                 # (1,1)

    dataset = TensorDataset(frames, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"✅ Loaded {len(dataset)} episodes → {len(loader)} batches")
    return loader, pred2frame
