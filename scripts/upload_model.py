#!/usr/bin/env python3
"""Upload model checkpoint to Hugging Face Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

# Set the token
os.environ["HF_TOKEN"] = "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"

# Model path (relative to project root)
project_root = Path(__file__).parent.parent
model_path = project_root / "experiments" / "radiant-forest-398" / "ckpt-step=053700-metric=0.00092727.pt"
repo_id = "chrisxx/pong"

api = HfApi(token=os.environ["HF_TOKEN"])

# Try to create the repository if it doesn't exist (as a model repo)
try:
    print(f"Creating repository {repo_id} if it doesn't exist...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=os.environ["HF_TOKEN"])
    print(f"✓ Repository ready")
except Exception as e:
    print(f"Note: {e}")

# Upload the model checkpoint
if model_path.exists():
    print(f"Uploading model from {model_path} to {repo_id}...")
    print(f"Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Use a cleaner filename in the repo
    repo_filename = "pytorch_model.pt"
    
    upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
    )
    print(f"✓ Uploaded model as {repo_filename}")
    
    # Also upload with original filename for reference
    upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=repo_id,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
    )
    print(f"✓ Uploaded model as {model_path.name}")
else:
    print(f"✗ Model file not found: {model_path}")

# Upload the config file
config_path = project_root / "configs" / "inference.yaml"
if config_path.exists():
    print(f"\nUploading config file from {config_path} to {repo_id}...")
    upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo=config_path.name,
        repo_id=repo_id,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
    )
    print(f"✓ Uploaded config as {config_path.name}")
else:
    print(f"\n✗ Config file not found: {config_path}")

print("\nUpload complete!")

