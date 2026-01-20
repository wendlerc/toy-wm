import math
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..eval import basic_control_dynamic, run_actions
from ..utils import log_video, get_muon, lr_lambda


def _frames_to_uint8(frames):
    if frames.ndim == 5:
        frames = frames[0]
    elif frames.ndim != 4:
        raise ValueError(f"Unexpected frame shape: {frames.shape}")

    frames = frames.detach().cpu()
    if frames.dtype == t.uint8:
        return frames

    vmin, vmax = frames.min().item(), frames.max().item()
    if vmax <= 1.05 and vmin >= 0.0:
        frames = (frames * 255.0).round()
    elif vmax <= 255.0 and vmin >= 0.0:
        frames = frames.round()
    else:
        frames = frames.clamp(0, 255).round()
    return frames.to(t.uint8)


def _frames_with_actions_image(frames, actions, max_frames=16, cols=4):
    frames_uint8 = _frames_to_uint8(frames)
    if actions.ndim == 3:
        actions = actions[0]
    actions = actions.detach().cpu()

    total = frames_uint8.shape[0]
    if total == 0:
        return None
    n = min(max_frames, total)
    if n < total:
        idx = t.linspace(0, total - 1, steps=n).long()
    else:
        idx = t.arange(total)

    frames_sel = frames_uint8[idx]
    actions_sel = actions[idx]

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        frame = frames_sel[i].permute(1, 2, 0).numpy()
        if frame.shape[2] == 1:
            ax.imshow(frame[:, :, 0], cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(frame)
        action = actions_sel[i]
        if action.numel() == 2:
            title = f"a=({int(action[0])},{int(action[1])})"
        else:
            title = f"a={action.tolist()}"
        ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    plt.close(fig)
    return wandb.Image(image)


def _paired_frames_with_actions_image(prev_frames, next_frames, actions, max_frames=16, cols=4):
    prev_uint8 = _frames_to_uint8(prev_frames)
    next_uint8 = _frames_to_uint8(next_frames)
    if actions.ndim == 3:
        actions = actions[0]
    actions = actions.detach().cpu()

    total = min(prev_uint8.shape[0], next_uint8.shape[0], actions.shape[0])
    if total == 0:
        return None
    n = min(max_frames, total)
    if n < total:
        idx = t.linspace(0, total - 1, steps=n).long()
    else:
        idx = t.arange(total)

    prev_sel = prev_uint8[idx]
    next_sel = next_uint8[idx]
    actions_sel = actions[idx]

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        prev_frame = prev_sel[i].permute(1, 2, 0).numpy()
        next_frame = next_sel[i].permute(1, 2, 0).numpy()
        pair = np.concatenate([prev_frame, next_frame], axis=1)
        if pair.shape[2] == 1:
            ax.imshow(pair[:, :, 0], cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(pair)
        action = actions_sel[i]
        if action.numel() == 2:
            title = f"a=({int(action[0])},{int(action[1])})"
        else:
            title = f"a={action.tolist()}"
        ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    plt.close(fig)
    return wandb.Image(image)


def _overlay_diff_with_actions_image(prev_frames, next_frames, actions, max_frames=16, cols=4):
    prev_uint8 = _frames_to_uint8(prev_frames)
    next_uint8 = _frames_to_uint8(next_frames)
    if actions.ndim == 3:
        actions = actions[0]
    actions = actions.detach().cpu()

    total = min(prev_uint8.shape[0], next_uint8.shape[0], actions.shape[0])
    if total == 0:
        return None
    n = min(max_frames, total)
    if n < total:
        idx = t.linspace(0, total - 1, steps=n).long()
    else:
        idx = t.arange(total)

    prev_sel = prev_uint8[idx]
    next_sel = next_uint8[idx]
    actions_sel = actions[idx]

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n:
            continue
        prev_frame = prev_sel[i].permute(1, 2, 0).numpy().astype(np.float32)
        next_frame = next_sel[i].permute(1, 2, 0).numpy().astype(np.float32)
        if prev_frame.shape[2] == 1:
            prev_frame = np.repeat(prev_frame, 3, axis=2)
            next_frame = np.repeat(next_frame, 3, axis=2)

        diff = np.abs(next_frame - prev_frame).mean(axis=2)
        denom = max(diff.max(), 1.0)
        diff_norm = (diff / denom).clip(0.0, 1.0)
        alpha = 0.7 * diff_norm[..., None]
        green = np.zeros_like(prev_frame)
        green[..., 1] = 255.0
        overlay = prev_frame * (1.0 - alpha) + green * alpha
        overlay = overlay.clip(0, 255).astype(np.uint8)

        ax.imshow(overlay)
        action = actions_sel[i]
        if action.numel() == 2:
            title = f"a=({int(action[0])},{int(action[1])})"
        else:
            title = f"a={action.tolist()}"
        ax.set_title(title, fontsize=8)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    plt.close(fig)
    return wandb.Image(image)


def train(model, action_model, action_decoder, 
          dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, 
          max_steps=1000, 
          warmup_steps=100,
          eval_each_n_steps = 100,
          clipping=True,
          checkpoint_manager=None,
          device="cuda", 
          dtype=t.float32):

    optimizer = get_muon(model, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay), 
                         action_model=action_model,
                         action_decoder=action_decoder)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=warmup_steps))

    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        model.train()
        log_dict = {}
        optimizer.zero_grad()
        try:
            frames, gt_actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, gt_actions = next(iterator)
            
        frames = frames.to(device).to(dtype)
        
        with t.autocast(device_type=device, dtype=dtype):
            # assuming true frame action sequence
            # (f_1, a_1), (f_2, a_2), ..., (f_dur, a_dur)
            # _, a_1, a_2, a_3, ..., a_{dur-1} \approx action_model(f_1, ..., f_dur)
            # note that the output for the first frame cannot contain the relevant information
            # latent action model
            actions, actions_cont, labels_pred, loss_vq = action_model(frames)
            # action decoder maps from current frame to next frame directly
            ts_dummy = t.zeros(actions.shape[0], actions.shape[1]-1, device=device, dtype=dtype)
            next_frames, _, _ = action_decoder(frames[:, :-1], actions[:, 1:], ts_dummy) # first action is throwaway
            loss_lam = F.mse_loss(next_frames.double(), frames[:, 1:].double(), reduction="mean")
            # dynamics model  
            ts = F.sigmoid(t.randn(actions.shape[0], actions.shape[1], device=device, dtype=dtype))
            x0 = frames
            z = t.randn_like(x0, device=device, dtype=dtype)
            vel_true = x0 - z
            x_t = x0 - ts[:, :, None, None, None] * vel_true
            # because action_pred is offset by 1, each frame gets the action of the previous frame as an input in this way
            vel_pred, _, _ = model(x_t, labels_pred, ts)
            loss_rf = F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean")
            loss = loss_lam + loss_rf + loss_vq
            
        loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(action_model.parameters(), 10.0)
            t.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        log_dict["loss_lam"] = loss_lam.item()
        log_dict["loss_rf"] = loss_rf.item()
        log_dict["loss_vq"] = loss_vq.item()
        log_dict["loss"] = loss.item()
        log_dict["lr"] = scheduler.get_last_lr()[0]
        if step % eval_each_n_steps == 0 and pred2frame is not None:
            # Log action histogram during eval (less frequently)
            #log_dict["codebook_grad_mean"] = action_model.learnt_actions.D.grad.mean().item()
            #log_dict["codebook_grad_std"] = action_model.learnt_actions.D.grad.std().item()
            log_dict["action_hist1"] = labels_pred[...,0].detach().cpu().numpy()
            log_dict["action_hist2"] = labels_pred[...,1].detach().cpu().numpy()

            vis_frames = pred2frame(frames)
            vis_next_frames = pred2frame(next_frames)
            vis_prev_frames = pred2frame(frames[:, :-1])
            grid_image = _frames_with_actions_image(vis_frames, labels_pred, max_frames=16, cols=4)
            if grid_image is not None:
                log_dict["frames_with_actions"] = grid_image
            decoder_pred_image = _frames_with_actions_image(vis_next_frames, labels_pred[:, 1:], max_frames=16, cols=4)
            if decoder_pred_image is not None:
                log_dict["action_decoder_pred_frames"] = decoder_pred_image
            paired_image = _paired_frames_with_actions_image(
                vis_prev_frames, vis_next_frames, labels_pred[:, 1:], max_frames=16, cols=4
            )
            if paired_image is not None:
                log_dict["action_decoder_pairs"] = paired_image
            overlay_image = _overlay_diff_with_actions_image(
                vis_prev_frames, vis_next_frames, labels_pred[:, 1:], max_frames=16, cols=4
            )
            if overlay_image is not None:
                log_dict["action_decoder_overlay_diff"] = overlay_image

            model.eval()
            # overwrite action embeddings with the ones from the codebook
            with t.no_grad():
                frames_actions = run_actions(model, labels_pred[0].unsqueeze(0))
                log_dict["control_actions"] = log_video(frames_actions, fps=30)
                frames_uncond = run_actions(model, t.zeros([1, 150, 2], dtype=t.int32, device=device))
                log_dict["control_uncond"] = log_video(frames_uncond, fps=30)
                # Compute the Cartesian product of two action codebooks, each of size (n_codes,)
                n_actions = model.action_emb1.weight.shape[0]
                idx1 = t.arange(n_actions, device=device)
                idx2 = t.arange(n_actions, device=device)
                grid1, grid2 = t.meshgrid(idx1, idx2, indexing='ij')
                actions_prod = t.stack([grid1.reshape(-1), grid2.reshape(-1)], dim=-1).unsqueeze(0)  # shape (1, n_codes1*n_codes2, 2)
                actions_prod = actions_prod.repeat_interleave(30, dim=1)
                frames_prod = run_actions(model, actions_prod.to(t.long).to(model.device))
                log_dict["control_prod"] = log_video(frames_prod, fps=30)
            
            checkpoint_manager.save(metric=loss.item(), 
                                    step=step, 
                                    model=model, 
                                    action_model=action_model,
                                    optimizer=optimizer, 
                                    scheduler=scheduler)
        wandb.log(log_dict)

    return model
