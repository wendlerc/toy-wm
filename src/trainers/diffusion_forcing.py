import torch as t
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..eval import basic_control
from ..utils import log_video, get_muon, lr_lambda


def train(model, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, 
          max_steps=1000, 
          warmup_steps=100,
          eval_each_n_steps = 500,
          clipping=True,
          checkpoint_manager=None,
          device="cuda", 
          dtype=t.float32):
    print(f"Using device: {device}, dtype: {dtype}")
    optimizer = get_muon(model, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=warmup_steps))

    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        model.train()
        log_dict = {}
        optimizer.zero_grad()
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, actions = next(iterator)
        
        mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
        actions[mask] = 0
        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
        
        if frames.shape[1] > model.n_window:
            print(f"Warning: frames.shape[1] > model.n_window, truncating to {model.n_window} frames")

        frames = frames[:, :model.n_window]
        actions = actions[:, :model.n_window]
        
        frames = frames.to(device).to(dtype)
        actions = actions.to(device)
        
        with t.autocast(device_type=device, dtype=dtype):
            ts = ts[:,:model.n_window]
            z = t.randn_like(frames, device=device, dtype=dtype)
            x0 = frames
            vel_true = x0 - z
            x_t = x0 - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _ = model(x_t, actions, ts)
            loss = F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean")
        
        loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        log_dict["loss"] = loss.item()
        log_dict["lr"] = scheduler.get_last_lr()[0]
        if step % eval_each_n_steps == 0 and pred2frame is not None:
            checkpoint_manager.save(metric=loss.item(), step=step, model=model, optimizer=optimizer, scheduler=scheduler)
            model.eval()
            # compute loss per noise level; TODO: do this smater as in min_rf repo
            noise_levels = [1., 0.75, 0.5, 0.25, 0.1, 0]
            noise_losses = []
            with t.no_grad():
                with t.autocast(device_type=device, dtype=dtype):
                    for noise_level in noise_levels:
                        z = t.randn_like(frames, device=device, dtype=dtype)
                        x0 = frames
                        vel_true = x0 - z
                        ts = noise_level * t.ones(frames.shape[0], frames.shape[1], device=device, dtype=dtype)
                        x_t = x0 - ts[:, :, None, None, None] * vel_true
                        vel_pred, _, _ = model(x_t, actions, ts)
                        noise_losses.append(F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean"))
                        log_dict[f"noise:{noise_level}"] = noise_losses[-1].item()

            if frames.shape[1] == 1: 
                with t.autocast(device_type=device, dtype=dtype):
                    z_sampled = sample(model, 
                                    t.randn_like(frames[:30], device=device, dtype=dtype), 
                                    actions[:30], num_steps=10)
                    z_sampled = z_sampled.permute(1, 0, 2, 3, 4)
            else:
                with t.autocast(device_type=device, dtype=dtype):
                    z_sampled = sample(model, t.randn_like(frames[:1], device=device, dtype=dtype), actions[:1], num_steps=10)
            frames_sampled = pred2frame(z_sampled)
            log_dict["sample"] = log_video(frames_sampled)
            log_dict["control"] = log_video(basic_control(model))
        wandb.log(log_dict)

    return model