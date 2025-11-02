import torch as t
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from pdb import set_trace

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

    return model