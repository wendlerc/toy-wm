from ..config import Config
from . import doom_pvp
from . import pong1m


def load_loader_and_pred2frame_from_config(config_path_or_cfg):
    if isinstance(config_path_or_cfg, str):
        print(f"loading {config_path_or_cfg}")
        cfg = Config.from_yaml(config_path_or_cfg)
    else:
        cfg = config_path_or_cfg
    c = cfg.dataset
    pad_height = cfg.model.patch_size > 1  # only pad when patch_size requires even height
    if c.dataset_id == "doom1p":
        kwargs = dict(batch_size=c.batch_size, duration=c.duration, fps=c.fps, debug=c.debug, num_workers=c.num_workers, shuffle=c.shuffle, pad_height=pad_height)
        shard_dir = getattr(c, "shard_dir", None)
        if shard_dir is not None:
            kwargs["shard_dir"] = shard_dir
        return doom_pvp.get_loader(**kwargs)
    elif c.dataset_id == "pong1p":
        return pong1m.get_loader(batch_size=c.batch_size, duration=c.duration, fps=c.fps, debug=c.debug)
    else:
        raise ValueError(f"Invalid dataset type: {c.dataset_id}")
