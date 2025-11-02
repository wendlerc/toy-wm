from .datasets.pong1m import get_loader
from .models.dit import get_model
from .trainers.rectified_flow import train
import wandb

import torch as t
t.set_float32_matmul_precision("high")

if __name__ == "__main__":
    wandb.init(project="toy-wm")
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

    loader, _, _ = get_loader(batch_size=32, duration=1, fps=7)
    frames, actions = next(iter(loader))
    height, width = frames.shape[-2:]
    model = get_model(height, width, n_window=7, patch_size=2, d_model=64, n_blocks=12)
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
    model = train(model, loader, lr1=0.01, lr2=1.5e-4, betas=(0.9, 0.95), weight_decay=1e-5, max_steps=26000)
    # 0.002, 3e-5, (0.9, 0.95), 1e-5, 26000 works ok

    import os
    from datetime import datetime

    # Create experiments directory if it doesn't exist
    exp_root = "experiments"
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(exp_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    t.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    print(f"Model saved to {os.path.join(save_dir, 'model.pt')}")