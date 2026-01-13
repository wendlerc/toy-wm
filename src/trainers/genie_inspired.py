import torch as t
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..eval import basic_control
from ..utils import log_video, get_muon, lr_lambda


def train(model, action_model, 
          dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, 
          max_steps=1000, 
          warmup_steps=100,
          eval_each_n_steps = 100,
          clipping=True,
          action_dropout=0.2,
          checkpoint_manager=None,
          device="cuda", 
          dtype=t.float32):

    optimizer = get_muon(model, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay), action_model=action_model)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=warmup_steps))

    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    for step in pbar:
        model.train()
        log_dict = {}
        optimizer.zero_grad()
        try:
            frames, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, _ = next(iterator)
            
        frames = frames.to(device).to(dtype)

        with t.autocast(device_type=device, dtype=dtype):
            # assuming true frame action sequence
            # (f_1, a_1), (f_2, a_2), ..., (f_dur, a_dur)
            # _, a_1, a_2, a_3, ..., a_{dur-1} \approx action_model(f_1, ..., f_dur)
            # note that the output for the first frame cannot contain the relevant information
            actions_pred, labels_pred, loss_vq = action_model(frames)
            # print(f'actions_pred.shape {actions_pred.shape}')
            actions = actions_pred[:, 1:]
            ts = F.sigmoid(t.randn(actions.shape[0], actions.shape[1], device=device, dtype=dtype))
            z = t.randn_like(frames[:,:-1], device=device, dtype=dtype)
            x0 = frames[:,:-1]
            vel_true = x0 - z
            x_t = x0 - ts[:, :, None, None, None] * vel_true
            vel_pred, _, _ = model(x_t, actions_pred[:, 1:], ts)
            loss_rf = F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean")
            loss = loss_rf + loss_vq
        
        loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(action_model.parameters(), 10.0)
            t.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
        log_dict["loss_rf"] = loss_rf.item()
        log_dict["loss_vq"] = loss_vq.item()
        log_dict["loss"] = loss.item()
        log_dict["lr"] = scheduler.get_last_lr()[0]
        if step % eval_each_n_steps == 0 and pred2frame is not None:
            # Log action histogram during eval (less frequently)
            log_dict["codebook_grad_mean"] = action_model.learnt_actions.grad.mean().item()
            log_dict["codebook_grad_std"] = action_model.learnt_actions.grad.std().item()
            log_dict["action_hist"] = labels_pred.detach().cpu().numpy()
            print("predicted action shapes", labels_pred.shape)
            print("predicted action labels", labels_pred)
            checkpoint_manager.save(metric=loss.item(), 
                                    step=step, 
                                    model=model, 
                                    action_model=action_model,
                                    optimizer=optimizer, 
                                    scheduler=scheduler)
            model.eval()

            if frames.shape[1] == 1: 
                with t.autocast(device_type=device, dtype=dtype):
                    z_sampled = sample(model, 
                                    t.randn_like(frames[:30], device=device, dtype=dtype), 
                                    actions[:30], num_steps=10)
                    z_sampled = z_sampled.permute(1, 0, 2, 3, 4)
            else:
                with t.autocast(device_type=device, dtype=dtype):
                    z_sampled = sample(model, t.randn_like(frames[:1, :-1], device=device, dtype=dtype), actions[:1], num_steps=10)
            frames_sampled = pred2frame(z_sampled)
            log_dict["sample"] = log_video(frames_sampled, fps=30)
        wandb.log(log_dict)

    return model