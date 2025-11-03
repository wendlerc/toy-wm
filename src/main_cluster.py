from .datasets.pong1m import get_loader
from .models.dit_dforce import get_model
from .trainers.diffusion_forcing import train
import wandb
import argparse
import os
from datetime import datetime
import torch as t
t.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    # 0.002, 3e-5, (0.9, 0.95), 1e-5, 26000 works ok
    parser.add_argument("--lr1", type=float, default=0.002)
    parser.add_argument("--lr2", type=float, default=3e-5)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.95))
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=26000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--n_window", type=int, default=7)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--noclip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    wandb.init(project="toy-wm")
    wandb.config.update(args)
    exp_root = "experiments"
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
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

    loader, _, _ = get_loader(batch_size=args.batch_size, duration=args.duration, fps=args.fps, debug=args.debug) # 7 was the max that does not go oom
    frames, actions = next(iter(loader))
    height, width = frames.shape[-2:]
    model = get_model(height, width, n_window=args.n_window, patch_size=args.patch_size, n_heads=args.n_heads,d_model=args.d_model, n_blocks=args.n_blocks, T=args.T)
    model = model.to(device)  # Move model to device

    # Apply torch compile for acceleration (PyTorch 2.0+)
    try:
        model = t.compile(model)
        print("Model compiled with torch.compile for acceleration.")
    except AttributeError:
        print("torch.compile is not available in this version of PyTorch; running without compilation.")

    #model = model.to(t.bfloat16)
    # Pass device to train if needed, or make sure trainer and dataloader use device
    # wandb.watch(model, log="all", log_freq=100)  # log_freq reduces logging overhead, log="all" avoids gradient tracking issues
    model = train(model, loader, 
                  lr1=args.lr1, lr2=args.lr2, betas=args.betas, 
                  weight_decay=args.weight_decay, max_steps=args.max_steps, clipping=~args.noclip)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")