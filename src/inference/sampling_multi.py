import torch as t

@t.no_grad()
def sample(v, z, actions, num_steps=10, cfg=0, negative_actions=None, cache=None):
    return sample_with_grad(v, z, actions, num_steps, cfg, negative_actions, cache=cache)

def sample_with_grad(v, z, actions, num_steps=10, cfg=0, negative_actions=None, cache=None):
    device = v.device
    ts = 1 - t.linspace(0, 1, num_steps+1, device=device)
    ts = 3*ts/(2*ts + 1)
    z_prev = z.clone()
    z_prev = z_prev.to(device)
    for i in range(len(ts)-1):
        t_cond = ts[i].repeat(z_prev.shape[0], 1)
        cached_k = None
        cached_v = None
        if cache is not None:
            cached_k, cached_v = cache.get()

        actions1 = actions.clone()
        actions2 = actions.clone()
        
        actions1[:, :, 0][actions1[:,:,0] != 0] = 0
        actions2[:, :, 1][actions2[:,:,1] != 0] = 0
        # later can be made more efficient by batching
        v_pred1, k_new1, v_new1 = v(z_prev.to(device), actions1.to(device), t_cond.to(device), cached_k=cached_k, cached_v=cached_v)    
        v_pred2, k_new2, v_new2 = v(z_prev.to(device), actions2.to(device), t_cond.to(device), cached_k=cached_k, cached_v=cached_v)        
        v_pred = 0.5 * v_pred1 + 0.5 * v_pred2
        k_new = 0.5 * k_new1 + 0.5 * k_new2 
        v_new = 0.5 * v_new1 + 0.5 * v_new2
        if cfg > 0:
            if negative_actions is not None:
                v_neg, _, _ = v(z_prev.to(device), negative_actions.to(device), t_cond.to(device), cached_k=cached_k, cached_v=cached_v)
            else:
                v_neg, _, _ = v(z_prev.to(device), t.zeros_like(actions, dtype=t.long, device=device), t_cond.to(device), cached_k=cached_k, cached_v=cached_v)
            v_pred = v_neg + cfg * (v_pred - v_neg)
        z_prev = z_prev + (ts[i] - ts[i+1])*v_pred 

    if cache is not None:
        cache.extend(k_new, v_new)

    return z_prev

def sample_video(model, actions, n_steps=4, cfg=0, negative_actions=None, clamp=True, cache=None):
    batch_size = actions.shape[0]
    num_actions = actions.shape[1]
    if cache is not None:
        cache.reset()
    else:
        cache = model.create_cache(batch_size)
    frames = t.randn(batch_size, num_actions, 3, 24, 24, device="cpu", dtype=model.dtype)
    for aidx in range(num_actions):
        noise=t.randn(batch_size, 1, 3, 24, 24, device=model.device, dtype=model.dtype)
        z = sample(model, noise, actions[:, aidx:aidx+1], num_steps=n_steps, cfg=cfg, negative_actions=negative_actions, cache=cache)
        frames[:, aidx:aidx+1] = z.detach().cpu()
        if clamp:
            frames = frames.clamp(-1, 1)
    return frames

