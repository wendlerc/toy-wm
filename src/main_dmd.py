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
    parser = argparse.ArgumentParser()    
    parser.add_argument("--student", type=str, default="configs/bigger_dmd.yaml")
    parser.add_argument("--teacher", type=str, default="configs/inference.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.student)
    teacher_cfg = Config.from_yaml(args.teacher)

    if "dtype" in cfg.train and cfg.train.dtype == "bf16":
        dtype = t.bfloat16
    else:
        dtype = t.float32
    cmodel = cfg.model
    ctrain = cfg.train

    assert teacher_cfg.model.checkpoint is not None, "DMD requires a checkpoint."
    
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    teacher_cfg_flat = OmegaConf.to_container(teacher_cfg, resolve=True)
    teacher_cfg_prefixed = {f"teacher_{k}": v for k, v in teacher_cfg_flat.items()}
    wandb.config.update(teacher_cfg_prefixed)
    exp_root = "experiments"
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
    if wandb.run.name is None:
        wandb.run.name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(exp_root, wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)

    if t.backends.mps.is_available():
        device = "mps"
    elif t.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loader, pred2frame = get_loader(batch_size=ctrain.batch_size, duration=ctrain.duration, fps=ctrain.fps, debug=ctrain.debug) # 7 was the max that does not go oom

    checkpoint_manager = CheckpointManager(save_dir, k=5, mode="min", metric_name="loss")
    model = train(args.student, args.teacher, loader, pred2frame=pred2frame,
                  lr1=ctrain.lr1, lr2=ctrain.lr2, betas=ctrain.betas, 
                  weight_decay=ctrain.weight_decay, max_steps=ctrain.max_steps,
                  clipping=not ctrain.noclip, checkpoint_manager=checkpoint_manager,
                  device=device, dtype=dtype, gradient_accumulation=ctrain.gradient_accumulation,
                  clamp_pred=ctrain.clamp_pred, warmup_steps=ctrain.warmup_steps,
                  n_fake_updates=ctrain.n_fake_updates)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")