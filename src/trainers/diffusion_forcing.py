import torch as t
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..eval import basic_control
from ..utils import log_video, get_muon, lr_lambda


@t.no_grad()
def _conditioned_ar_dit(model, first_frame, actions, n_steps=6, cfg=1.0):
    """Generate autoregressively from a real first frame with real actions (DiT)."""
    B = first_frame.shape[0]
    C, H, W = model.in_channels, model.height, model.width
    cache = model.create_cache(2 * B)
    # Seed cache with real first frame (ts=0)
    uncond = model.action_emb.unconditional_action
    if actions.dim() == 3:
        neg = uncond.to(actions).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
    else:
        neg = t.full_like(actions[:, :1], fill_value=int(uncond))
    init_act = t.zeros_like(actions[:, :1])
    act_batch = t.cat([init_act, init_act], dim=0)
    frame_batch = first_frame.repeat(2, 1, 1, 1, 1).to(model.device).to(model.dtype)
    ts_zero = t.zeros(2 * B, 1, device=model.device, dtype=model.dtype)
    ck, cv = cache.get()
    _, k_new, v_new = model(frame_batch, act_batch, ts_zero, cached_k=ck, cached_v=cv)
    cache.extend(k_new, v_new)
    # Generate remaining frames
    T = actions.shape[1]
    out = [first_frame.to(model.device).to(model.dtype)]
    for tidx in range(T):
        noise = t.randn(B, 1, C, H, W, device=model.device, dtype=model.dtype)
        z = sample(model, noise, actions[:, tidx:tidx+1],
                   num_steps=n_steps, cfg=cfg, cache=cache)
        out.append(z.detach())
    return t.cat(out, dim=1)


def train(model, dataloader,
          pred2frame=None,
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01,
          max_steps=1000,
          warmup_steps=100,
          eval_each_n_steps = 500,
          clipping=True,
          action_dropout=0.2,
          checkpoint_manager=None,
          device="cuda",
          dtype=t.float32):
    # Unwrap compiled / DDP model for attribute access
    eval_model = model
    for attr in ("_orig_mod",):
        eval_model = getattr(eval_model, attr, eval_model)

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

        # Action dropout — handle both discrete (B, T) and continuous (B, T, D)
        uncond = eval_model.action_emb.unconditional_action
        if actions.dim() == 3:
            # Doom: (B, T, 15) — mask whole timesteps
            mask_ts = t.rand(actions.shape[0], actions.shape[1], device=device, dtype=dtype) <= action_dropout
            mask = mask_ts.unsqueeze(-1).expand_as(actions)
            actions[mask] = uncond.to(actions).unsqueeze(0).expand(mask_ts.sum().item(), -1).reshape(-1)
        else:
            # Pong: (B, T) — mask individual timesteps
            mask = t.rand_like(actions, device=device, dtype=dtype) <= action_dropout
            actions[mask] = uncond

        ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))

        if frames.shape[1] > eval_model.n_window:
            print(f"Warning: frames.shape[1] > model.n_window, truncating to {eval_model.n_window} frames")

        frames = frames[:, :eval_model.n_window]
        actions = actions[:, :eval_model.n_window]

        frames = frames.to(device).to(dtype)
        actions = actions.to(device)

        # Save clean copies for AR-vs-GT
        frames_clean = frames.clone()
        actions_clean = actions.clone()

        with t.autocast(device_type=device, dtype=dtype):
            ts = ts[:,:eval_model.n_window]
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
            t.cuda.empty_cache()
            # compute loss per noise level
            noise_levels = [1., 0.75, 0.5, 0.25, 0.1, 0]
            noise_losses = []
            eval_frames = frames[:2]
            eval_actions = actions[:2]
            with t.no_grad():
                with t.autocast(device_type=device, dtype=dtype):
                    for noise_level in noise_levels:
                        z = t.randn_like(eval_frames, device=device, dtype=dtype)
                        x0 = eval_frames
                        vel_true = x0 - z
                        ts = noise_level * t.ones(eval_frames.shape[0], eval_frames.shape[1], device=device, dtype=dtype)
                        x_t = x0 - ts[:, :, None, None, None] * vel_true
                        vel_pred, _, _ = model(x_t, eval_actions, ts)
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
            log_dict["sample"] = log_video(frames_sampled, fps=30)
            frames_control = basic_control(model, pred2frame)
            log_dict["control"] = log_video(frames_control, fps=30)
            # AR-vs-GT comparison
            with t.no_grad():
                gt_ctx = frames_clean[:1, :1]
                gt_actions = actions_clean[:1, 1:eval_model.n_window]
                n_ar = gt_actions.shape[1]
                pred_ar = _conditioned_ar_dit(model, gt_ctx, gt_actions, n_steps=6, cfg=1.0)
                gt_clip = frames_clean[:1, :n_ar+1]
                pred_rgb = pred2frame(pred_ar)
                gt_rgb = pred2frame(gt_clip)
                sidebyside = t.cat([gt_rgb, pred_rgb], dim=-1)
                log_dict["ar_vs_gt"] = log_video(sidebyside, fps=30)
            # Offload VAE if using latent space (doom) to free VRAM
            if eval_model.in_channels != 3:
                from ..datasets.doom_pvp import offload_vae
                offload_vae(device)
        wandb.log(log_dict)

    return model
