#!/usr/bin/env python3
"""Download model checkpoint from Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Set the token (optional, can also use HF_TOKEN env var or login)
os.environ["HF_TOKEN"] = "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"

# Model path (relative to project root)
project_root = Path(__file__).parent.parent
model_dir = project_root / "experiments" / "bigger_30frame_causal"
repo_id = "chrisxx/pong"

# Model filename to download (using the cleaner name)
model_filename = "pytorch_model.pt"

# Create model directory if it doesn't exist
model_dir.mkdir(parents=True, exist_ok=True)
print(f"Model directory: {model_dir.absolute()}")

print(f"Downloading model from {repo_id} to {model_dir.absolute()}...")

try:
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    # Verify the file was downloaded to the expected location
    expected_path = model_dir / model_filename
    if expected_path.exists():
        file_size_mb = expected_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded {model_filename} to {expected_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        print(f"⚠ Downloaded to {downloaded_path}, but expected {expected_path}")
except Exception as e:
    print(f"✗ Error downloading model: {e}")

# Download the config file
config_filename = "inference.yaml"
config_dir = project_root / "configs"
config_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading config file from {repo_id} to {config_dir.absolute()}...")

try:
    downloaded_config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(config_dir),
        local_dir_use_symlinks=False,
    )
    # Verify the file was downloaded to the expected location
    expected_config_path = config_dir / config_filename
    if expected_config_path.exists():
        print(f"✓ Downloaded {config_filename} to {expected_config_path}")
    else:
        print(f"⚠ Downloaded to {downloaded_config_path}, but expected {expected_config_path}")
except Exception as e:
    print(f"✗ Error downloading config file: {e}")

print("\nDownload complete!")

