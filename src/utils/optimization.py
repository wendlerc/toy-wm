from muon import SingleDeviceMuonWithAuxAdam
import math

def get_muon(model, lr1, lr2, betas, weight_decay, action_model=None):
    body_weights = list(model.blocks.parameters())
    body_ids = {id(p) for p in body_weights}
    other_weights = [p for p in model.parameters() if id(p) not in body_ids]

    hidden_weights = [p for p in body_weights if p.ndim >= 2]
    hidden_gains_biases = [p for p in body_weights if p.ndim < 2]
    nonhidden_params = list(other_weights)

    if action_model is not None:
        body_weights2 = list(action_model.dit.blocks.parameters())
        body_ids2 = {id(p) for p in body_weights2}
        other_weights2 = [p for p in action_model.parameters() if id(p) not in body_ids2]

        hidden_weights2 = [p for p in body_weights2 if p.ndim >= 2]
        hidden_gains_biases2 = [p for p in body_weights2 if p.ndim < 2]
        nonhidden_params2 = list(other_weights2)

        hidden_weights += hidden_weights2
        hidden_gains_biases += hidden_gains_biases2
        nonhidden_params += nonhidden_params2

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