#!/usr/bin/env python3
"""
Helper script to set up and deploy Neural Pong to Hugging Face Spaces
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return output."""
    print(f"💻 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def check_dependencies():
    """Check if required tools are installed."""
    print("🔍 Checking dependencies...")
    
    # Check git
    try:
        run_command("git --version", check=True)
        print("✅ Git is installed")
    except:
        print("❌ Git is not installed. Please install it first.")
        return False
    
    # Check git-lfs
    try:
        run_command("git lfs version", check=True)
        print("✅ Git LFS is installed")
    except:
        print("❌ Git LFS is not installed. Install it with: git lfs install")
        return False
    
    # Check huggingface-cli
    try:
        run_command("huggingface-cli --version", check=True)
        print("✅ Hugging Face CLI is installed")
    except:
        print("⚠️  Hugging Face CLI not found. Install it with: pip install huggingface-hub")
        print("   (Optional but recommended)")
    
    return True

def create_space(username, space_name):
    """Create a new HF Space using the CLI."""
    print(f"\n🚀 Creating Space: {username}/{space_name}")
    
    # Check if logged in
    result = run_command("huggingface-cli whoami", check=False)
    if result.returncode != 0:
        print("⚠️  Not logged in to Hugging Face. Please login:")
        run_command("huggingface-cli login")
    
    # Create space (this will fail if space exists, which is fine)
    print("\nNote: Manual creation recommended at https://huggingface.co/new-space")
    print(f"Create a Gradio Space with GPU support")
    input("Press Enter once you've created the space...")

def prepare_files(source_dir, target_dir, checkpoint_path):
    """Copy necessary files to target directory."""
    print(f"\n📦 Preparing files in {target_dir}")
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "app.py",
        "requirements.txt",
        ".gitattributes",
    ]
    
    # Copy to README.md from HF_README.md
    if (source_dir / "HF_README.md").exists():
        shutil.copy2(source_dir / "HF_README.md", target_dir / "README.md")
        print("✅ Copied HF_README.md -> README.md")
    
    # Copy individual files
    for file in files_to_copy:
        src = source_dir / file
        if src.exists():
            shutil.copy2(src, target_dir / file)
            print(f"✅ Copied {file}")
    
    # Copy directories
    dirs_to_copy = ["src", "configs"]
    for dir_name in dirs_to_copy:
        src_dir = source_dir / dir_name
        dst_dir = target_dir / dir_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir, 
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
            print(f"✅ Copied {dir_name}/")
    
    # Handle checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n📊 Checkpoint found: {checkpoint_path}")
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        if size_mb > 100:
            print("\n⚠️  Large checkpoint detected!")
            print("   Options:")
            print("   1. Upload to HF Model Hub (Recommended)")
            print("   2. Include in Space with Git LFS")
            choice = input("   Your choice (1/2): ").strip()
            
            if choice == "1":
                print("\n📤 Upload checkpoint to HF Hub with:")
                print(f"""
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="{checkpoint_path}",
    path_in_repo="model.pt",
    repo_id="YOUR_USERNAME/neural-pong-checkpoint",
    repo_type="model"
)
                """)
                print("Then update configs/inference.yaml with:")
                print('  checkpoint: "hf://YOUR_USERNAME/neural-pong-checkpoint/model.pt"')
            else:
                # Copy checkpoint
                rel_path = Path(checkpoint_path).relative_to(source_dir)
                dst_checkpoint = target_dir / rel_path
                dst_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint_path, dst_checkpoint)
                print(f"✅ Copied checkpoint to {rel_path}")
    
    print("\n✅ Files prepared successfully!")

def init_git_repo(target_dir, space_url):
    """Initialize git repo and set up LFS."""
    print(f"\n🔧 Setting up Git repository")
    os.chdir(target_dir)
    
    # Check if already a git repo
    if os.path.exists(".git"):
        print("✅ Git repository already exists")
    else:
        run_command("git init")
        print("✅ Initialized git repository")
    
    # Set up Git LFS
    run_command("git lfs install")
    print("✅ Git LFS initialized")
    
    # Add remote if provided
    if space_url:
        # Check if remote exists
        result = run_command("git remote get-url origin", check=False)
        if result.returncode != 0:
            run_command(f"git remote add origin {space_url}")
            print(f"✅ Added remote: {space_url}")
        else:
            print(f"✅ Remote already configured: {space_url}")

def commit_and_push(target_dir):
    """Commit files and push to HF."""
    print("\n📤 Committing and pushing to Hugging Face")
    os.chdir(target_dir)
    
    # Add files
    run_command("git add .")
    
    # Commit
    commit_msg = """Initial commit: Neural Pong diffusion game

- Gradio interface for real-time gameplay
- Diffusion transformer with KV-caching
- Keyboard controls for paddle movement
- Model: 8-layer DiT trained on 1M Pong frames
"""
    run_command(f'git commit -m "{commit_msg}"', check=False)
    
    # Push
    print("\n🚀 Pushing to Hugging Face...")
    result = run_command("git push -u origin main", check=False)
    
    if result.returncode != 0:
        print("⚠️  Push failed. You may need to:")
        print("   1. Set up authentication: huggingface-cli login")
        print("   2. Manually push: cd", target_dir, "&& git push -u origin main")

def main():
    """Main setup flow."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║     🎮 Neural Pong - Hugging Face Space Setup 🚀         ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get user input
    print("\n📝 Configuration:")
    source_dir = input("Source directory (toy-wm path) [.]: ").strip() or "."
    source_dir = os.path.abspath(source_dir)
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)
    
    username = input("Your Hugging Face username: ").strip()
    space_name = input("Space name [neural-pong]: ").strip() or "neural-pong"
    
    target_dir = input(f"Target directory [{space_name}]: ").strip() or space_name
    target_dir = os.path.abspath(target_dir)
    
    # Find checkpoint
    checkpoint_path = None
    config_file = Path(source_dir) / "configs" / "inference.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
            if 'model' in config and 'checkpoint' in config['model']:
                rel_ckpt = config['model']['checkpoint']
                checkpoint_path = Path(source_dir) / rel_ckpt
                if not checkpoint_path.exists():
                    print(f"⚠️  Checkpoint not found: {checkpoint_path}")
                    checkpoint_path = None
    
    # Confirm
    print("\n📋 Summary:")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Space: {username}/{space_name}")
    print(f"  Checkpoint: {checkpoint_path if checkpoint_path else 'Not found'}")
    
    if input("\nProceed? (y/n): ").lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Create space
    create_space(username, space_name)
    
    # Prepare files
    prepare_files(source_dir, target_dir, checkpoint_path)
    
    # Set up git
    space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    init_git_repo(target_dir, space_url)
    
    # Ask about pushing
    if input("\nPush to Hugging Face now? (y/n): ").lower() == 'y':
        commit_and_push(target_dir)
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                    ✅ Setup Complete!                     ║
╚═══════════════════════════════════════════════════════════╝

Next steps:
1. Go to https://huggingface.co/spaces/{username}/{space_name}/settings
2. Select GPU hardware (T4 small recommended)
3. Wait for build to complete (~5 minutes)
4. Play your game at: https://huggingface.co/spaces/{username}/{space_name}

Files are in: {target_dir}

If you haven't pushed yet:
  cd {target_dir}
  git push -u origin main

Enjoy your Neural Pong game! 🎮
    """)

if __name__ == "__main__":
    main()




