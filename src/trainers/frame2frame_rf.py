import torch as t
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..utils import log_video, get_muon, lr_lambda


def train(model, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, 
          max_steps=1000, 
          warmup_steps=100,
          clipping=True,
          checkpoint_manager=None,
          device="cuda", dtype=t.float32):
    print(f"Using device: {device}, dtype: {dtype}")
    optimizer = get_muon(model, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=warmup_steps))

    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        log_dict = {}
        optimizer.zero_grad()
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, actions = next(iterator)
        
        mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
        actions[mask] = 0
        
        frames = frames.to(device).to(dtype)
        actions = actions.to(device)
        
        with t.autocast(device_type=device, dtype=dtype):
            z = frames[:, :-1]
            ts = F.sigmoid(t.randn(z.shape[0], z.shape[1], device=device, dtype=dtype))
            x0 = frames[:, 1:]
            vel_true = x0 - z
            x_t = x0 - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _ = model(x_t, actions[:,:-1], ts)
            loss = F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean")
        
        loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        log_dict["loss"] = loss.item()
        log_dict["lr"] = scheduler.get_last_lr()[0]
        if step % 100 == 0 and pred2frame is not None:
            checkpoint_manager.save(metric=loss.item(), step=step, model=model, optimizer=optimizer, scheduler=scheduler)
            # compute loss per noise level
            noise_levels = [1., 0.75, 0.5, 0.25, 0.1, 0]
            noise_losses = []
            with t.no_grad():
                with t.autocast(device_type=device, dtype=dtype):
                    for noise_level in noise_levels:
                        z = frames[:, :-1]
                        x0 = frames[:, 1:]
                        vel_true = x0 - z
                        ts = noise_level * t.ones(z.shape[0], z.shape[1], device=device, dtype=dtype)
                        x_t = x0 - ts[:, :, None, None, None] * vel_true
                        vel_pred, _, _ = model(x_t, actions[:, :-1], ts)
                        noise_losses.append(F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean"))
                        log_dict[f"noise:{noise_level}"] = noise_losses[-1].item()

            if frames.shape[1] == 1: 
                raise ValueError(f"frames.shape[1] has to be greater than 1.")
            else:
                with t.autocast(device_type=device, dtype=dtype):
                    z_sampled = sample(model, frames[:1, :-1], actions[:1,:-1], num_steps=10)
            frames_sampled = pred2frame(z_sampled)
            log_dict["sample"] = log_video(frames_sampled)
        wandb.log(log_dict)

    return model