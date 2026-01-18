import torch as t
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from functools import partial

from ..inference.sampling import sample
from ..eval import basic_control_dynamic, run_actions
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

    tmp_head = nn.Linear(320, 4)
    tmp_head.to(device).to(dtype)
    optimizer = get_muon(model, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay), action_model=action_model)
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, max_steps=max_steps, warmup_steps=warmup_steps))

    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    first = True
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
            actions_pred, actions_cont, labels_pred, loss_vq = action_model(frames)
            pred_gt_actions = tmp_head(actions_pred[:, 1:])
            gt_actions = gt_actions[:, :-1] 
            #ce_loss = F.cross_entropy(pred_gt_actions.reshape(-1, 4), gt_actions.reshape(-1).to(t.long).to(device))
            loss = loss_vq #+ ce_loss
            # lets also compute accuracy
            #accuracy = (pred_gt_actions.reshape(-1, 4).argmax(dim=1) == gt_actions.reshape(-1).to(t.long).to(device)).float().mean()
            #log_dict["accuracy"] = accuracy.item()
            #log_dict["ce_loss"] = ce_loss.item()
            # print(f'actions_pred.shape {actions_pred.shape}')
            if False:
                actions = actions_pred[:, 1:] # actually the way things line up in the mmdit we don't need to omit the first one
                ts = F.sigmoid(t.randn(actions.shape[0], actions.shape[1], device=device, dtype=dtype))
                x0 = frames[:,1:]
                z = frames[:,:-1]
                pred, _, _ = model(z, actions, 0*ts)
                loss_rf = F.mse_loss(pred.double(), x0.double(), reduction="mean")
            if True:
                actions = actions_pred
                ts = F.sigmoid(t.randn(actions.shape[0], actions.shape[1], device=device, dtype=dtype))
                x0 = frames
                z = t.randn_like(x0, device=device, dtype=dtype)
                vel_true = x0 - z
                x_t = x0 - ts[:, :, None, None, None] * vel_true
                # because action_pred is offset by 1, each frame gets the action of the previous frame as an input in this way
                vel_pred, _, _ = model(x_t, actions_pred, ts)
                loss_rf = F.mse_loss(vel_pred.double(), vel_true.double(), reduction="mean")
            loss += loss_rf
            
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
            #log_dict["codebook_grad_mean"] = action_model.learnt_actions.D.grad.mean().item()
            #log_dict["codebook_grad_std"] = action_model.learnt_actions.D.grad.std().item()
            log_dict["action_hist"] = labels_pred.detach().cpu().numpy()
            checkpoint_manager.save(metric=loss.item(), 
                                    step=step, 
                                    model=model, 
                                    action_model=action_model,
                                    optimizer=optimizer, 
                                    scheduler=scheduler)
            model.eval()
            # overwrite action embeddings with the ones from the codebook
            with t.no_grad():
                codebook = action_model.learnt_actions
                model.action_emb.weight.copy_(t.cat([action_model.unconditional_action.view(1, -1), codebook.project_out(codebook.codebook)], dim=0))
                # do some sanity checks; note that because we are comparing action embeddings before and after a gradient step, differences are supposed to be small but not 0.
                labels_match = labels_pred.to(t.long)
                embed_rows = model.action_emb(labels_match)
                actions_match = actions_pred
                cos_sim = F.cosine_similarity(actions_match, embed_rows, dim=-1)
                log_dict["action_embed_cosine_mean"] = cos_sim.mean().item()
                log_dict["action_embed_cosine_std"] = cos_sim.std().item()
                log_dict["action_embed_mse"] = F.mse_loss(actions_match, embed_rows).item()
            # call basic control dynamic with the 10 most common actions using bincount
            labels_np = labels_pred.detach().cpu().numpy().astype(np.int64)
            most_common_actions = np.bincount(labels_np.ravel()).argsort()[-10:][::-1]
            frames_control = basic_control_dynamic(model, most_common_actions.tolist())
            log_dict["control_most_common"] = log_video(frames_control, fps=30)
            frames_actions = run_actions(model, labels_pred[0].unsqueeze(0))
            log_dict["control_actions"] = log_video(frames_actions, fps=30)
            frames_uncond = run_actions(model, t.zeros([1, 150], dtype=t.int32, device=device))
            log_dict["control_uncond"] = log_video(frames_uncond, fps=30)
        wandb.log(log_dict)

    return model