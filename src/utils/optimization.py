from muon import SingleDeviceMuonWithAuxAdam
import math

def get_muon(model, lr1, lr2, betas, weight_decay):
    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]

    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]
    nonhidden_params = list(other_weights)

    param_groups = [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=lr1,
            weight_decay=weight_decay,
        ),
        dict(
            params=hidden_gains_biases + nonhidden_params,
            use_muon=False,
            lr=lr2,
            betas=betas,
            weight_decay=weight_decay,
        ),
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    return optimizer


def lr_lambda(current_step, max_steps, warmup_steps=100):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))