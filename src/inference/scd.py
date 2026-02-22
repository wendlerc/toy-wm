import torch as t

@t.no_grad()
def sample(v, z, frames, actions, num_steps=10, cfg=1.0, cache=None):
    return sample_with_grad(v, z, frames, actions, num_steps, cfg, cache=cache)

def sample_with_grad(v, z, frames, actions, num_steps=10, cfg=1.0, cache=None):
    device = v.device
    ts = 1 - t.linspace(0, 1, num_steps+1, device=device)
    ts = 3*ts/(2*ts + 1)
    z_prev = z.clone()
    z_prev = z_prev.to(device)
    encoder_output = None
    for i in range(len(ts)-1):
        t_cond = ts[i].repeat(z_prev.shape[0], 1)
        cached_k = None
        cached_v = None
        if cache is not None:
            cached_k, cached_v = cache.get()

        negative_actions = t.zeros_like(actions, dtype=t.long, device=device)
        
        actions_batch = t.cat([actions, negative_actions], dim=0)
        z_batch = z_prev.repeat(2, 1, 1, 1, 1)
        t_batch = t_cond.repeat(2, 1)
        frames_batch = frames.repeat(2, 1, 1, 1, 1)
        v_pred, encoder_output, k_new, v_new = v(z_for_decoder=z_batch,
                                                z_for_encoder=frames_batch, 
                                                actions=actions_batch, 
                                                ts=t_batch, 
                                                encoder_output=encoder_output, 
                                                cached_k=cached_k, 
                                                cached_v=cached_v)            
        v_pred, v_neg = v_pred.chunk(2, dim=0)
        v_pred = v_neg + cfg * (v_pred - v_neg)
        z_prev = z_prev + (ts[i] - ts[i+1])*v_pred 
        if cache is not None and k_new is not None:
            cache.extend(k_new, v_new)

    return z_prev

def sample_video(model, actions, n_steps=4, cfg=1.0, clamp=True, cache=None):
    # TODO: revisit this and check for 1 off errors
    batch_size = actions.shape[0]
    num_actions = actions.shape[1]
    if cache is not None:
        cache.reset()
    else:
        cache = model.create_cache(2*batch_size)
    frames = t.randn(batch_size, num_actions+1, 3, 24, 24, device=model.device, dtype=model.dtype)
    actions = t.cat([t.zeros_like(actions[:, :1]), actions], dim=1)
    frames[:, 0] = 0
    for aidx in range(num_actions-2):
        noise=t.randn(batch_size, 1, 3, 24, 24, device=model.device, dtype=model.dtype)
        z = sample(model, noise, frames[:, aidx:aidx+1], actions[:, aidx:aidx+1], num_steps=n_steps, cfg=cfg, cache=cache)
        frames[:, aidx+1:aidx+2] = z
        if clamp:
            frames = frames.clamp(-1, 1)
    return frames.detach().cpu()

