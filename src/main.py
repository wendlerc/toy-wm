from .datasets.pong1m import get_loader
from .models.dit_dforce import get_model
from .trainers.diffusion_forcing import train
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

    loader,pred2frame = get_loader(batch_size=ctrain.batch_size, duration=ctrain.duration, fps=ctrain.fps, debug=ctrain.debug) # 7 was the max that does not go oom
    frames, actions = next(iter(loader))
    height, width = frames.shape[-2:]
    model = get_model(height, width, 
                    n_window=cmodel.n_window, 
                    patch_size=cmodel.patch_size, 
                    n_heads=cmodel.n_heads,d_model=cmodel.d_model, 
                    n_blocks=cmodel.n_blocks, 
                    T=cmodel.T, 
                    in_channels=cmodel.in_channels,
                    bidirectional=cmodel.bidirectional)
    if cmodel.checkpoint is not None:
        print(f"Loading model from {cmodel.checkpoint}")
        state_dict = t.load(cmodel.checkpoint, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "_orig_mod." in list(state_dict.keys())[0]:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("_orig_mod.")}
        model.load_state_dict(state_dict)
        print(f"Model loaded from {cmodel.checkpoint}")
    else:
        print("No checkpoint found")
    model = model.to(device)  # Move model to device
    #model = model.to(t.bfloat16)
    # Apply torch compile for acceleration (PyTorch 2.0+)
    if not cmodel.nocompile:
        try:
            model = t.compile(model)
            print("Model compiled with torch.compile for acceleration.")
        except AttributeError:
            print("torch.compile is not available in this version of PyTorch; running without compilation.")

    wandb.watch(model, log="all", log_freq=100)  # log_freq reduces logging overhead, log="all" avoids gradient tracking issues
    checkpoint_manager = CheckpointManager(save_dir, k=5, mode="min", metric_name="loss")

    model = train(model, loader, pred2frame=pred2frame,
                  lr1=ctrain.lr1, lr2=ctrain.lr2, betas=ctrain.betas, 
                  weight_decay=ctrain.weight_decay, max_steps=ctrain.max_steps, 
                  clipping=not ctrain.noclip, checkpoint_manager=checkpoint_manager)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")