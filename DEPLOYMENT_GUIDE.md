# 🚀 Deployment Guide: Neural Pong to Hugging Face Spaces

This guide will walk you through deploying your Neural Pong game to Hugging Face Spaces.

## Prerequisites

- A Hugging Face account ([sign up here](https://huggingface.co/join))
- Git and Git LFS installed
- Your trained model checkpoint
- CUDA GPU (provided by HF Spaces)

## Quick Start

### 1. Create a Hugging Face Space

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login
huggingface-cli login
```

Go to https://huggingface.co/spaces and click "Create new Space":
- **Name:** `neural-pong` (or your choice)
- **License:** MIT
- **SDK:** Gradio
- **Hardware:** T4 small GPU (or better)

### 2. Prepare Your Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/neural-pong
cd neural-pong

# Copy application files from your toy-wm directory
cp /path/to/toy-wm/app.py .
cp /path/to/toy-wm/requirements.txt .
cp /path/to/toy-wm/HF_README.md README.md
cp /path/to/toy-wm/.gitattributes .
cp /path/to/toy-wm/.gitignore_hf .gitignore

# Copy source code
cp -r /path/to/toy-wm/src .
cp -r /path/to/toy-wm/configs .
```

### 3. Handle Model Checkpoint

You have **two options** for the checkpoint:

#### Option A: Upload to HF Model Hub (Recommended - Faster Loading)

```python
from huggingface_hub import HfApi, create_repo

# Create model repo
repo_id = "YOUR_USERNAME/neural-pong-checkpoint"
create_repo(repo_id, repo_type="model", exist_ok=True)

# Upload checkpoint
api = HfApi()
api.upload_file(
    path_or_fileobj="experiments/radiant-forest-398/ckpt-step=053700-metric=0.00092727.pt",
    path_in_repo="model.pt",
    repo_id=repo_id,
)

print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")
```

Then update `configs/inference.yaml`:

```yaml
model:
  # Change this line:
  # checkpoint: "experiments/radiant-forest-398/ckpt-step=053700-metric=0.00092727.pt"
  # To this:
  checkpoint: "hf://YOUR_USERNAME/neural-pong-checkpoint/model.pt"
```

And update `app.py` to handle HF Hub paths:

```python
# Add at the top of initialize_model():
from huggingface_hub import hf_hub_download

# Before loading the model:
if checkpoint_path.startswith("hf://"):
    # Parse HF path: hf://username/repo/file
    parts = checkpoint_path[5:].split("/")
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:])
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
```

#### Option B: Include in Space with Git LFS

```bash
# Initialize Git LFS
git lfs install
git lfs track "experiments/**/*.pt"

# Copy checkpoint
mkdir -p experiments/radiant-forest-398
cp /path/to/checkpoint.pt experiments/radiant-forest-398/ckpt-step=053700-metric=0.00092727.pt

# Verify LFS tracking
git lfs ls-files
```

### 4. Handle Dataset (Optional)

If your model needs the dataset files at inference:

```bash
# Option 1: Include small dataset files
mkdir -p datasets/pong1M
cp /path/to/toy-wm/datasets/pong1M/*.npy datasets/pong1M/

# Option 2: Load from HF Datasets (if available)
# Modify app.py to download from datasets hub
```

If the dataset is only needed for the `pred2frame` function, you may be able to skip it or refactor the code.

### 5. Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: Neural Pong diffusion game

- Gradio interface for real-time gameplay
- Diffusion transformer with KV-caching
- Keyboard controls for paddle movement
- Model: 8-layer DiT trained on 1M Pong frames
"

# Push to HF Space
git push
```

### 6. Configure Space Settings

1. Go to your Space on HF: `https://huggingface.co/spaces/YOUR_USERNAME/neural-pong`
2. Click **Settings**
3. Under **Space hardware**, select:
   - **T4 small** (basic, ~$0.60/hour)
   - **T4 medium** (better, ~$1.50/hour)
   - **A10G small** (fast, ~$3.15/hour)
4. Set **Visibility** to Public or Private
5. (Optional) Enable **Persistent Storage** if you need to cache things
6. (Optional) Enable **Always On** for 24/7 availability

### 7. Monitor Deployment

Watch the build logs:
- Go to your Space URL
- Check the "App logs" section
- Look for "✅ Neural Pong is ready!"

Initial deployment takes ~5-10 minutes:
- Installing dependencies (~2-3 min)
- Downloading checkpoint (~1-2 min)
- Loading model to GPU (~1-2 min)
- Warmup (~30 sec)

## Testing Your Space

Once deployed:

1. Open your Space URL
2. Wait for "Ready! Click 'Start Game' to begin."
3. Click **"Start Game"**
4. Use ↑/↓ or W/S keys to play!

## Troubleshooting

### Build Fails: "Could not find CUDA"

**Solution:** Make sure you selected GPU hardware in Space settings.

### "Checkpoint not found"

**Solution:** 
- Check that the path in `configs/inference.yaml` is correct
- If using Option A, verify the HF Hub path is correct
- If using Option B, ensure Git LFS tracked the file: `git lfs ls-files`

### Slow Loading

**Solution:**
- Option A (HF Hub) is faster for checkpoint loading
- Consider using a smaller checkpoint for demo
- Check Space logs for bottlenecks

### "Model loading..." Forever

**Solution:**
- Check app logs for errors
- Verify all `src/` files are included
- Ensure `configs/inference.yaml` is present

### Keyboard Not Working

**Solution:**
- Click on the game area to ensure browser focus
- Check browser console for JavaScript errors
- Try using Chrome or Firefox

## Optimization Tips

### Reduce Startup Time

```python
# In app.py, add caching:
from huggingface_hub import snapshot_download

# Pre-download checkpoint at startup
if not os.path.exists("checkpoint_cache"):
    snapshot_download(
        "YOUR_USERNAME/neural-pong-checkpoint",
        local_dir="checkpoint_cache",
        local_dir_use_symlinks=False
    )
```

### Improve Performance

1. **Reduce diffusion steps:** Change `n_steps=4` to `n_steps=2` in app.py
2. **Lower FPS:** Change `fps=4` to `fps=2` for more compute per frame
3. **Use better GPU:** Upgrade to A10G for 2-3x speedup

### Add Analytics

```python
# Track usage with HF Analytics
from huggingface_hub import HfApi

api = HfApi()
api.like("spaces/YOUR_USERNAME/neural-pong")  # Track likes
```

## Sharing Your Space

Once deployed:

1. **Share the URL:** `https://huggingface.co/spaces/YOUR_USERNAME/neural-pong`
2. **Embed in website:**
   ```html
   <iframe src="https://YOUR_USERNAME-neural-pong.hf.space" 
           width="850" height="900"></iframe>
   ```
3. **Add to your HF profile** as a pinned Space
4. **Tweet about it!** 🐦 Tag @huggingface

## Cost Estimation

Approximate costs for different hardware tiers:

| GPU | Cost/Hour | Startup Time | FPS | Recommended For |
|-----|-----------|--------------|-----|-----------------|
| T4 small | $0.60 | ~60s | 3-4 | Development/Demo |
| T4 medium | $1.50 | ~45s | 4-5 | Public Demo |
| A10G small | $3.15 | ~30s | 8-10 | Production |

**Tip:** Use "Sleep after inactivity" to reduce costs when not in use.

## Next Steps

- **Add game scoring:** Track ball bounces and display score
- **Multiple difficulty levels:** Adjust AI opponent speed
- **Record gameplay:** Save generated sequences as videos
- **Compare with real Pong:** Side-by-side comparison
- **Interactive training:** Let users add their gameplay to training data

## Support

- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- HF Forums: https://discuss.huggingface.co/
- This project: [Create an issue]

---

**Good luck with your deployment! 🚀**

If you run into issues, share your Space URL and error logs in the HF forums for help.




