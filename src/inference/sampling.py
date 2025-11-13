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

        v_pred, k_new, v_new = v(z_prev.to(device), actions.to(device), t_cond.to(device), cached_k=cached_k, cached_v=cached_v)

        if i == len(ts)-2:
            cache.extend(k_new, v_new)

        if cfg > 0:
            if cache is not None:
                raise NotImplementedError("this is not implemented yet")
            if negative_actions is not None:
                v_neg = v(z_prev.to(device), negative_actions.to(device), t_cond.to(device))
            else:
                v_neg = v(z_prev.to(device), t.zeros_like(actions, dtype=t.long, device=device), t_cond.to(device))
            v_pred = v_neg + cfg * (v_pred - v_neg)
        z_prev = z_prev + (ts[i] - ts[i+1])*v_pred 
    return z_prev