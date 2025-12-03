#!/usr/bin/env python3
"""Download pong model from Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Model path (relative to project root)
project_root = Path(__file__).parent.parent
model_path = project_root / "experiments" / "pong"
repo_id = "chrisxx/pong"

# File to download
filename = "model.pt"

# Create model directory if it doesn't exist
model_path.mkdir(parents=True, exist_ok=True)
print(f"Model directory: {model_path.absolute()}")

print(f"Downloading model from {repo_id} to {model_path.absolute()}...")

output_path = model_path / filename
print(f"Downloading {filename}...")
try:
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )
    # Verify the file was downloaded to the expected location
    expected_path = model_path / filename
    if expected_path.exists():
        print(f"✓ Downloaded {filename} to {expected_path}")
    else:
        print(f"⚠ Downloaded to {downloaded_path}, but expected {expected_path}")
except Exception as e:
    print(f"✗ Error downloading {filename}: {e}")

print("Download complete!")

