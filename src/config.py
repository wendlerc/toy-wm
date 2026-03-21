from dataclasses import dataclass, field, fields as dc_fields, MISSING
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
    trainer_id : str = "diffusion_forcing"
    lr1 : float = 0.002
    lr2 : float = 3e-5
    betas : tuple = (0.9, 0.95)
    weight_decay : float = 1e-5
    max_steps : int = 26000
    warmup_steps : int = 100
    noclip : bool = False
    dtype : str = "bf16"
    action_dropout : float = 0.2


@dataclass
class WANDBConfig:
    name : str = "toy-wm"
    project : str = None
    run_name : str = None

@dataclass
class DatasetConfig:
    dataset_id: str = "pong1p"
    num_workers: int = 8
    batch_size: int = 64
    duration: int = 1
    fps: int = 30
    shuffle: bool = True
    debug: bool = False
    shard_dir: Optional[str] = None

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        # Merge each sub-config with dataclass defaults so partial
        # yaml sections (e.g. dataset with only 3 of 8 fields) work.
        sub_configs = {
            'model': TransformerConfig,
            'train': TrainingConfig,
            'wandb': WANDBConfig,
            'dataset': DatasetConfig,
        }
        for key, dc_cls in sub_configs.items():
            defaults = {}
            for f in dc_fields(dc_cls):
                if f.default is not MISSING:
                    defaults[f.name] = f.default
                elif f.default_factory is not MISSING:
                    defaults[f.name] = f.default_factory()
            section = raw_cfg.get(key, {}) or {}
            defaults.update(section)
            raw_cfg[key] = defaults

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))
