#!/usr/bin/env python3
"""
Quick script to upload your model checkpoint to Hugging Face Hub
This is the recommended approach for large checkpoints (>100MB)
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, login
except ImportError:
    print("❌ huggingface_hub not installed")
    print("Install it with: pip install huggingface-hub")
    sys.exit(1)

def upload_checkpoint(checkpoint_path, repo_name, username=None):
    """Upload a checkpoint to HF Model Hub."""
    
    # Verify checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"📊 Checkpoint: {checkpoint_path}")
    print(f"   Size: {size_mb:.1f} MB")
    
    # Login if needed
    try:
        api = HfApi()
        user_info = api.whoami()
        if username is None:
            username = user_info['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print("⚠️  Not logged in to Hugging Face")
        print("Logging in now...")
        login()
        user_info = api.whoami()
        username = user_info['name']
    
    # Create repo ID
    repo_id = f"{username}/{repo_name}"
    print(f"\n🚀 Uploading to: {repo_id}")
    
    # Create repo
    try:
        url = create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository ready: {url}")
    except Exception as e:
        print(f"⚠️  Error creating repo: {e}")
        print("Continuing with upload...")
    
    # Upload file
    print(f"\n📤 Uploading {checkpoint_path.name}...")
    print("   (This may take a few minutes depending on file size)")
    
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="model.pt",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Upload complete!")
        
        # Print next steps
        hf_path = f"hf://{repo_id}/model.pt"
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║                  ✅ Upload Successful!                    ║
╚═══════════════════════════════════════════════════════════╝

Your checkpoint is now available at:
  https://huggingface.co/{repo_id}

Next steps:

1. Update your configs/inference.yaml:
   
   model:
     checkpoint: "{hf_path}"

2. Update your app.py to handle HF Hub paths.
   Add this code in the initialize_model() function:

   # Before loading the model
   if checkpoint_path.startswith("hf://"):
       from huggingface_hub import hf_hub_download
       parts = checkpoint_path[5:].split("/")
       repo_id = f"{{parts[0]}}/{{parts[1]}}"
       filename = "/".join(parts[2:])
       checkpoint_path = hf_hub_download(
           repo_id=repo_id, 
           filename=filename
       )

3. Deploy to Hugging Face Spaces!

Enjoy! 🎮
        """)
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║       📤 Upload Checkpoint to Hugging Face Hub 🤗        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Get checkpoint path
    default_ckpt = "experiments/radiant-forest-398/ckpt-step=053700-metric=0.00092727.pt"
    checkpoint_path = input(f"Checkpoint path [{default_ckpt}]: ").strip()
    if not checkpoint_path:
        checkpoint_path = default_ckpt
    
    # Get repo name
    default_repo = "neural-pong-checkpoint"
    repo_name = input(f"Repository name [{default_repo}]: ").strip()
    if not repo_name:
        repo_name = default_repo
    
    # Optional: specify username
    username_input = input("Your HF username (leave empty for auto-detect): ").strip()
    username = username_input if username_input else None
    
    # Confirm
    print(f"\n📋 Summary:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Repository: {username or '[auto]'}/{repo_name}")
    
    if input("\nProceed? (y/n): ").lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Upload
    upload_checkpoint(checkpoint_path, repo_name, username)

if __name__ == "__main__":
    main()




