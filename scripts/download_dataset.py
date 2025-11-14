#!/usr/bin/env python3
"""Download pong1M dataset from Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Set the token (optional, can also use HF_TOKEN env var or login)
os.environ["HF_TOKEN"] = "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"

# Dataset path (relative to project root)
project_root = Path(__file__).parent.parent
dataset_path = project_root / "datasets" / "pong1M"
repo_id = "chrisxx/pong"

# Files to download
files_to_download = ["actions.npy", "frames.npy", "rewards.npy"]

# Create dataset directory if it doesn't exist
dataset_path.mkdir(parents=True, exist_ok=True)
print(f"Dataset directory: {dataset_path.absolute()}")

print(f"Downloading dataset from {repo_id} to {dataset_path.absolute()}...")

for filename in files_to_download:
    output_path = dataset_path / filename
    print(f"Downloading {filename}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            local_dir=str(dataset_path),
            local_dir_use_symlinks=False,
        )
        # Verify the file was downloaded to the expected location
        expected_path = dataset_path / filename
        if expected_path.exists():
            print(f"✓ Downloaded {filename} to {expected_path}")
        else:
            print(f"⚠ Downloaded to {downloaded_path}, but expected {expected_path}")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print("Download complete!")

