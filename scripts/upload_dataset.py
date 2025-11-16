#!/usr/bin/env python3
"""Upload pong1M dataset to Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

# Set the token
os.environ["HF_TOKEN"] = "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"

# Dataset path (relative to project root)
dataset_path = Path(__file__).parent.parent / "datasets" / "pong1M"
repo_id = "chrisxx/pong"

# Files to upload
files_to_upload = ["actions.npy", "frames.npy", "rewards.npy"]

api = HfApi(token=os.environ["HF_TOKEN"])

# Try to create the repository if it doesn't exist
try:
    print(f"Creating repository {repo_id} if it doesn't exist...")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=os.environ["HF_TOKEN"])
    print(f"✓ Repository ready")
except Exception as e:
    print(f"Note: {e}")

print(f"Uploading dataset from {dataset_path} to {repo_id}...")

for filename in files_to_upload:
    file_path = dataset_path / filename
    if file_path.exists():
        print(f"Uploading {filename}...")
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        print(f"✓ Uploaded {filename}")
    else:
        print(f"✗ File not found: {file_path}")

print("Upload complete!")










