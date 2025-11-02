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

@t.no_grad()
def sample(v, z, frames, actions, num_steps=50, uniform=False):
    device = v.device
    if uniform: 
        ts = 1 - t.linspace(0, 1, num_steps, device=device)
    else:
        ts = 1 - F.sigmoid(t.randn(num_steps, device=device).msort())
    z_prev = z.clone()
    z_prev = z_prev.to(device)
    for i in tqdm(range(len(ts)-1)):
        t_cond = ts[i].repeat(z_prev.shape[0], 1)
        z_prev = z_prev + (ts[i] - ts[i+1])*v(z_prev.to(device), frames[:, :-1].to(device), actions.to(device), t_cond.to(device)) 
    return z_prev

@t.no_grad():
def log_gif(z, tag="generated_gif"):
    """
    Create a gif from z[0] and log it to wandb.

    Args:
        z: torch.Tensor, shape (1, num_frames, C, H, W)
        tag: str, wandb image label
    """
    # Convert frames to numpy: (num_frames, H, W, C)
    frames_np = z[0].permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(frames_np[0])
    ax.set_title('Frame 0')

    def animate(i):
        im.set_data(frames_np[i])
        ax.set_title(f'Frame {i}')
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, frames=frames_np.shape[0],
        interval=200, blit=True, repeat=True
    )

    # Save animation to buffer as gif and upload to wandb
    buf = io.BytesIO()
    ani.save(buf, format='gif')
    buf.seek(0)

    wandb.log({tag: wandb.Video(buf, format="gif")})

    plt.close(fig)

def train(model, dataloader, lr=1e-4, weight_decay=1e-4, max_steps=1000):

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
            z_sampled = sample(model, t.randn_like(frames, device=device, dtype=dtype), frames, actions)
            log_gif(z_sampled, tag=f"generated_gif_{step}")

    return model