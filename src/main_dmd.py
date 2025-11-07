from .datasets.pong1m import get_loader
from .models.dit_dforce import get_model
from .trainers.dmd import train
import wandb
import argparse
import os
from datetime import datetime
import torch as t
from .config import Config
from omegaconf import OmegaConf
from .utils.checkpoint import CheckpointManager

t.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    # 0.002, 3e-5, (0.9, 0.95), 1e-5, 26000 works ok
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    dtype = t.float32
    cmodel = cfg.model
    ctrain = cfg.train
    assert cmodel.checkpoint is not None, "DMD requires a checkpoint."

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    exp_root = "experiments"
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
    if wandb.run.name is None:
        wandb.run.name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(exp_root, wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)
    # Detect MPS (Apple Silicon) or CUDA if available
    if t.backends.mps.is_available():
        device = t.device("mps")
        print("Using device: MPS")
    elif t.cuda.is_available():
        device = t.device("cuda")
        print("Using device: CUDA")
    else:
        device = t.device("cpu")
        print("Using device: CPU")

    loader, pred2frame = get_loader(batch_size=ctrain.batch_size, duration=ctrain.duration, fps=ctrain.fps, debug=ctrain.debug) # 7 was the max that does not go oom

    checkpoint_manager = CheckpointManager(save_dir, k=5, mode="min", metric_name="loss")
    p_pretrain = ctrain.p_pretrain if "p_pretrain" in ctrain else 1.0
    model = train(args.config, loader, pred2frame=pred2frame,
                  lr1=ctrain.lr1, lr2=ctrain.lr2, betas=ctrain.betas, 
                  weight_decay=ctrain.weight_decay, max_steps=ctrain.max_steps, p_pretrain=p_pretrain,
                  clipping=not ctrain.noclip, checkpoint_manager=checkpoint_manager,
                  device=device, dtype=dtype)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")