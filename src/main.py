from .datasets.pong1m import get_loader
from .models.dit import get_model
import wandb
import argparse
import os
from datetime import datetime
import torch as t
from .config import Config
from omegaconf import OmegaConf
from .utils.checkpoint import CheckpointManager, load_model_from_config
from .trainers.diffusion_forcing import train

t.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cmodel = cfg.model
    ctrain = cfg.train

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
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
    frames, actions = next(iter(loader))
    model = load_model_from_config(args.config)

    dtype = t.bfloat16 if ctrain.dtype == "bf16" else t.float32
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device) 
    model = model.to(dtype)

    if not cmodel.nocompile:
        try:
            model = t.compile(model)#, mode="max-autotune")
            print("Model compiled with torch.compile for acceleration.")
        except AttributeError:
            print("torch.compile is not available in this version of PyTorch; running without compilation.")

    wandb.watch(model, log="all", log_freq=100)  # log_freq reduces logging overhead, log="all" avoids gradient tracking issues
    checkpoint_manager = CheckpointManager(save_dir, k=5, mode="min", metric_name="loss")
    p_pretrain = ctrain.p_pretrain if "p_pretrain" in ctrain else 1.0
    model = train(model, loader, pred2frame=pred2frame,
                  lr1=ctrain.lr1, lr2=ctrain.lr2, betas=ctrain.betas, 
                  weight_decay=ctrain.weight_decay, max_steps=ctrain.max_steps,
                  clipping=not ctrain.noclip, checkpoint_manager=checkpoint_manager,
                  warmup_steps=ctrain.warmup_steps, device=device, dtype=dtype)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")