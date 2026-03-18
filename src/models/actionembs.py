import torch as t
import torch.nn as nn
from ..nn.pe import NumericEncoding
from jaxtyping import Float, Int
from torch import Tensor

PONG_ACTION_NAMES = ["uncond", "noop", "up", "down"]


class Pong1P(nn.Module):
    def __init__(self, d_model, n_actions=4):
        """
        unconditional, noop, up, down
        """
        super().__init__()
        self.action_emb = nn.Embedding(n_actions, d_model)

    def format_action(self, action) -> str:
        """Format a single action for display (scalar int)."""
        return PONG_ACTION_NAMES[action.item()]

    def forward(self, actions: Int[Tensor, "batch dur"]):
        return self.action_emb(actions)

    @property
    def unconditional_action(self):
        return int(0)

    @property
    def basic_control_actions(self):
        return t.tensor(30*[1] + 60*[2] + 60*[3] + 30*[0], dtype=t.int32, device=self.device).unsqueeze(0)

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def dtype(self):
        return self.parameters().__next__().dtype


DOOM_ACTION_NAMES = ["uncond", "MF", "MB", "MR", "ML", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "ATK", "SPD", "TURN"]
DOOM_ACTION_DISPLAY_NAMES = ["uncond", "Fwd", "Back", "Right", "Left", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "Atk", "Run", "Turn"]


class Doom1P(nn.Module):
    """
    - Dim 0: uncond (other dims should be all zero)
    - Dims 1-13: binary {0, 1}
    - Dim 14 (turn): {-12.5, -11.25, ..., 0, ..., 11.25, 12.5} — 21 values, step size 1.25
    Actions: ["uncond",  "MF", # MOVE_FORWARD "MB", # MOVE_BACKWARD "MR", # MOVE_RIGHT "ML", # MOVE_LEFT "W1", # SELECT_WEAPON1 "W2", # SELECT_WEAPON2 "W3", # SELECT_WEAPON3 "W4", # SELECT_WEAPON4 "W5", # SELECT_WEAPON5 "W6", # SELECT_WEAPON6 "W7", # SELECT_WEAPON7 "ATK", # ATTACK "SPD", # SPEED "TURN",# TURN_LEFT_RIGHT_DELTA ]
    """
    def __init__(self, d_model, d_turn=32, n_max_turn_emb=100):
        """
        Args:
            d_model: the dimension of the model
            d_turn: the dimension of the turn embedding (before it goes into linear)
            n_max_turn_emb: the maximum value of the turn embedding (for the numeric encoding that takes the real number x and creates bunch of sin (x), cos(x) for it)
        """
        super().__init__()
        self.action_emb = nn.Linear(14 + d_turn, d_model, bias=False)
        self.angle_emb = NumericEncoding(dim=d_turn, n_max=n_max_turn_emb)
        self.n_max_turn_emb = n_max_turn_emb
        self.register_buffer("uncond_action", t.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=t.float32))

    def format_action(self, action) -> str:
        """Format a single action for display ((15,) vector)."""
        a = action.detach().cpu()
        if a.numel() == 15:
            if a[0].item() != 0:
                return "uncond"
            parts = [DOOM_ACTION_DISPLAY_NAMES[i] for i in range(1, 14) if a[i].item() != 0]
            if a[14].item() != 0:
                parts.append(f"T:{a[14].item():+.1f}")
            return "+".join(parts) if parts else "Idle"
        return str(a.tolist())

    def forward(self, actions: Float[Tensor, "batch dur act"]):
        dtype = self.action_emb.weight.dtype
        actions = actions.to(dtype)
        angle = self.angle_emb(((12.5 + actions[:, :, -1])/25 * (self.n_max_turn_emb - 1)).int())
        actions = actions.clone()  # avoid in-place on potential autograd input
        actions[actions[:, :, 0] == 1] = t.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=actions.device)
        angle[actions[:, :, 0] == 1] = 0
        return self.action_emb(t.cat([actions[:, :, :-1], angle], dim=-1))

    @property
    def unconditional_action(self):
        return self.uncond_action

    @property
    def basic_control_actions(self):
        actions = []
        for i in range(14):
            action = [0] * 15
            action[i] = 1
            actions += 30*[action]

        action = [0] * 15
        action[14] = -6.25
        actions += 30*[action]
        action = [0] * 15
        action[14] = 6.25
        actions += 30*[action]
        return t.tensor(actions, dtype=self.dtype, device=self.device).unsqueeze(0)

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def dtype(self):
        return self.parameters().__next__().dtype
