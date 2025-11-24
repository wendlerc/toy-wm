import wandb
import torch as t

def log_video(z, fps=5):
    """
    Log a video to wandb from a tensor of frames.
    Ensures that wandb receives a uint8 tensor in [0, 255].
    z: (B, T, C, H, W) or (T, C, H, W) in [0, 255] uint8 or [0, 1] or [0, 255] float
    """

    if z.ndim == 5:
        frames = z[0]        # take first in batch
    elif z.ndim == 4:
        frames = z
    else:
        raise ValueError(f"Unexpected shape: {z.shape}")

    # Check and convert to uint8 in [0, 255]
    if frames.dtype == t.uint8:
        frames_uint8 = frames.cpu().numpy()
    else:
        # It's a float or int; check the value range and convert
        vmin, vmax = frames.min().item(), frames.max().item()
        frames_proc = frames
        if vmax <= 1.05 and vmin >= 0.0:
            # Probably [0, 1] float
            frames_proc = (frames * 255.0).round()
        elif vmax <= 255.0 and vmin >= 0.0:
            # Possibly a float in [0,255], so just round
            frames_proc = frames.round()
        else:
            # Unusual, just clip to [0,255]
            frames_proc = frames.clamp(0, 255).round()
        frames_uint8 = frames_proc.to(t.uint8).cpu().numpy()

    return wandb.Video(frames_uint8, fps=fps, format="mp4")