from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from omegaconf import OmegaConf

@dataclass
class TransformerConfig:
    model_id : str = None
    width : int = 24
    height : int = 24
    T : int = 1000
    in_channels : int = 3
    n_window : int = 7
    patch_size : int = 2
    n_heads : int = 4
    d_model : int = 64
    n_blocks : int = 12
    n_heads : int = 12
    d_model : int = 384
    patch_size : int = 1
    bidirectional : bool = True
    nocompile : bool = False
    checkpoint : str = None


@dataclass
class TrainingConfig:
    lr1 : float = 0.002
    lr2 : float = 3e-5
    betas : tuple = (0.9, 0.95)
    weight_decay : float = 1e-5
    max_steps : int = 26000
    batch_size : int = 32
    noclip : bool = False
    duration : int = 1
    fps : int = 7
    in_channels : int = 3
    debug : bool = False


@dataclass
class WANDBConfig:
    name : str = "toy-wm"
    project : str = None
    run_name : str = None 

@dataclass
class ActionModelConfig:
    model_id : str = "action_dit"
    n_actions : int = 6
    d_actions : int = 64
    beta : float = 0.25
    action_dropout : float = 0.0
    checkpoint : str = None

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig
    action_model: Optional[ActionModelConfig] = None

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)
        
        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))
