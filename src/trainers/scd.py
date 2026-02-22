import torch as t
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from functools import partial
import random

from ..inference.scd import sample_video
from ..eval import basic_control
from ..utils import log_video, get_muon, lr_lambda


def train(model, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, 
          max_steps=1000, 
          warmup_steps=100,
          eval_each_n_steps = 500,
          clipping=True,
          action_dropout = 0.2,
          first_dropout = 0.1,
          checkpoint_manager=None,
          device="cuda", 
          dtype=t.float32):
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

        assert frames.shape[1] == model.n_window + 1, "frames.shape[1] must be equal to model.n_window + 1 for this trainer"
        # frames are       x_1, x_2, ...,        x_w, x_w+1
        # actions are a_0, a_1, a_2, ..., a_w-1, a_w
        frames_enc = frames[:, :-1]
        actions_enc = actions[:, 1:]
        frames_dec = frames[:, 1:]

        mask = t.rand_like(actions_enc, device=device, dtype=dtype) <= action_dropout
        actions_enc[mask] = 0
        if random.random() < first_dropout:
            frames_enc[:, 0] *= 0
            actions_enc[:, 0] = 0

        frames_enc = frames_enc.to(device).to(dtype)
        actions_enc = actions_enc.to(device)
        frames_dec = frames_dec.to(device).to(dtype)
        ts = F.sigmoid(t.randn(frames_dec.shape[0], frames_dec.shape[1], device=device, dtype=dtype))
                
        with t.autocast(device_type=device, dtype=dtype):
            z = t.randn_like(frames_dec, device=device, dtype=dtype)
            x0 = frames_dec
            vel_true = x0 - z
            x_t = x0 - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _, _ = model(z_for_decoder=x_t, ts=ts, z_for_encoder=frames_enc, actions=actions_enc)
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
            log_dict["fwd"] = log_video(pred2frame(x_t[:1] + vel_pred[:1]), fps=30)
            frames_control = basic_control(model, sample_video=sample_video)
            log_dict["control"] = log_video(frames_control, fps=30)
        wandb.log(log_dict)

    return model