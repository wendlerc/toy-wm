from .datasets.pong1m import get_loader
from .models.dit import get_model
from .trainers.rectified_flow import train
import wandb

import torch as t
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

    loader = get_loader(duration=1, fps=12)
    frames, actions = next(iter(loader))
    height, width = frames.shape[-2:]
    model = get_model(height, width, patch_size=4)
    model = model.to(device)  # Move model to device
    #model = model.to(t.bfloat16)
    # Pass device to train if needed, or make sure trainer and dataloader use device
    wandb.watch(model)
    model = train(model, loader, lr=1e-4)

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