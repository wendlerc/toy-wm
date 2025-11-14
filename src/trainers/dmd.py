import torch as t
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from functools import partial

from ..utils import log_video, load_model_from_config, get_muon, lr_lambda
from ..inference import sample


def train(student_cfg, teacher_cfg, dataloader, 
          pred2frame=None, 
          lr1=0.02, lr2=3e-4, betas=(0.9, 0.95), weight_decay=0.01, max_steps=1000, 
          clipping=True,
          checkpoint_manager=None,
          n_fake_updates=5, 
          device=None, dtype=None, 
          clamp_pred = False,
          gradient_accumulation=1,
          warmup_steps=100):


    true_v = load_model_from_config(teacher_cfg)
    fake_v = load_model_from_config(teacher_cfg)
    gen = load_model_from_config(student_cfg)

    true_v.to(device).to(dtype)
    fake_v.to(device).to(dtype)
    gen.to(device).to(dtype)
    for p in true_v.parameters():
        p.requires_grad = False

    true_v = t.compile(true_v)
    fake_v = t.compile(fake_v)
    gen = t.compile(gen)
    
    fake_opt = get_muon(fake_v, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))
    gen_opt = get_muon(gen, float(lr1), float(lr2), (float(betas[0]), float(betas[1])), float(weight_decay))

    fake_sched = t.optim.lr_scheduler.LambdaLR(fake_opt, partial(lr_lambda, max_steps=n_fake_updates*max_steps//gradient_accumulation, warmup_steps=warmup_steps*n_fake_updates//gradient_accumulation))
    gen_sched = t.optim.lr_scheduler.LambdaLR(gen_opt, partial(lr_lambda, max_steps=max_steps//gradient_accumulation, warmup_steps=warmup_steps//gradient_accumulation))
    iterator = iter(dataloader)
    pbar = tqdm(range(max_steps))
    step_fake = 0
    for sidx, step in enumerate(pbar):
        #set_trace()
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            frames, actions = next(iterator)
        
        mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
        actions[mask] = 0
        if frames.shape[1] > gen.n_window:
            print(f"Warning: frames.shape[1] > gen.n_window, truncating to {gen.n_window} frames")
            frames = frames[:, :gen.n_window]
            actions = actions[:, :gen.n_window]
        frames = frames.to(device)
        actions = actions.to(device)
        with t.autocast(device_type=device, dtype=dtype):
            z = t.randn_like(frames, device=device, dtype=dtype)
            gen_ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            x_inp = frames - gen_ts[:,:,None,None,None]*(frames - z)
            v_pred, _, _ = gen(x_inp, actions, gen_ts)
            x_pred = x_inp + gen_ts[:,:,None,None,None]*v_pred
            if clamp_pred:
                x_pred = t.clamp(x_pred, -1.0, 1.0)
            # sample fresh noise
            z = t.randn_like(frames, device=device, dtype=dtype)
            v_pred = x_pred - z
            # compute dmd gradient
            ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
            x_t = x_pred - ts[:,:,None,None,None]*v_pred # does it matter that we reuse the noise from the generation step here?
            x_t_nograd = x_t.detach()
            fake_vel, _, _ = fake_v(x_t_nograd, actions, ts)
            real_vel, _, _ = true_v(x_t_nograd, actions, ts)
            
            gen_loss = 0.5*((x_pred.double() - x_pred.detach().double() - (real_vel.detach().double() - fake_vel.detach().double()))**2).mean()
        
        if sidx > 0:
            gen_loss.backward()
            if clipping:
                t.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
            if (sidx + 1) % gradient_accumulation == 0:
                wandb.log({"gen_loss": gen_loss.item()})
                wandb.log({"gen_lr": gen_sched.get_last_lr()[0]})
                gen_opt.step()
                gen_sched.step()
                gen_opt.zero_grad()
        
        # update fake_v
        fake_loss = F.mse_loss(fake_vel.double(), v_pred.detach().double())
        fake_loss.backward()
        if clipping:
            t.nn.utils.clip_grad_norm_(fake_v.parameters(), 10.0)
        step_fake += 1
        if (step_fake + 1) % gradient_accumulation == 0:
            wandb.log({"fake_loss": fake_loss.item()})
            wandb.log({"fake_lr": fake_sched.get_last_lr()[0]})
            fake_opt.step()
            fake_sched.step()
            fake_opt.zero_grad()
        for _ in range(n_fake_updates-1): #TODO this needs to be fixed to be the same as above
            try:
                frames, actions = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                frames, actions = next(iterator)
            
            mask = t.rand_like(actions, device=device, dtype=dtype) < 0.2
            actions[mask] = 0
            actions = actions.to(device)
            frames = frames.to(device)
            # gen sample
            with t.autocast(device_type=device, dtype=dtype):
                gen_ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
                z = t.randn_like(frames, device=device, dtype=dtype)
                x_inp = frames - gen_ts[:,:,None,None,None]*(frames - z)
                with t.no_grad():
                    v_pred, _, _ = gen(x_inp, actions, gen_ts)
                x_pred = x_inp + gen_ts[:,:,None,None,None]*v_pred
                if clamp_pred:
                    x_pred = t.clamp(x_pred, -1.0, 1.0)
                # sample fresh noise
                z = t.randn_like(frames, device=device, dtype=dtype)
                v_pred = x_pred - z         
                ts = F.sigmoid(t.randn(frames.shape[0], frames.shape[1], device=device, dtype=dtype))
                x_t = x_pred - ts[:,:,None,None,None]*v_pred 
                x_t_nograd = x_t.detach()
                fake_vel, _, _ = fake_v(x_t_nograd, actions, ts)
                fake_loss = F.mse_loss(fake_vel.double(), v_pred.detach().double())
            fake_loss.backward()
            step_fake += 1
            if clipping:
                t.nn.utils.clip_grad_norm_(fake_v.parameters(), 10.0)
            if (step_fake + 1) % gradient_accumulation == 0:
                wandb.log({"fake_loss": fake_loss.item()})
                wandb.log({"fake_lr": fake_sched.get_last_lr()[0]})
                fake_opt.step()
                fake_sched.step()
                fake_opt.zero_grad()

        pbar.set_postfix_str(f'loss_gen {gen_loss.item():.4f} loss_fake {fake_loss.item():.4f}')

        if step % 100 == 0 and pred2frame is not None:
            with t.no_grad():
                teacher_sample = sample(true_v, x_inp, actions, num_steps=4)
                student_teacher_loss = F.mse_loss(teacher_sample, x_pred)
                eval_loss = F.mse_loss(x_pred, frames)
                teacher_loss = F.mse_loss(teacher_sample, frames)
                wandb.log({"eval_loss":eval_loss.item()})
                wandb.log({"teacher_loss":teacher_loss.item()})
                wandb.log({"student_teacher_loss":student_teacher_loss.item()})
                checkpoint_manager.save(metric=eval_loss.item(), step=step, model=gen, optimizer=gen_opt, scheduler=None)
                frame_preds = x_pred
                frames_sampled = pred2frame(frame_preds.detach().cpu())
                log_video(frames_sampled)
    return gen