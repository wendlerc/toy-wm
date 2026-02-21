from .diffusion_forcing import train as diffusion_forcing
from .scd import train as scd
from ..config import Config


def load_train_fct_from_config(config_path: str):
    print(f"loading {config_path}")
    ctrain = Config.from_yaml(config_path).train
    if ctrain.trainer_id == "diffusion_forcing":
        return diffusion_forcing
    elif ctrain.trainer_id == "scd":
        return scd
    else:
        raise ValueError(f"Invalid trainer type: {ctrain.trainer_id}")