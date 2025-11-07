import wandb

def log_video(z, fps=5):
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

    frames_uint8 = frames.byte().cpu().numpy()
    wandb.log({"sample": wandb.Video(frames_uint8, fps=fps, format="mp4")})