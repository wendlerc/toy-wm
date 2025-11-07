# 📦 Hugging Face Space Deployment Package - Summary

This package contains everything you need to deploy your Neural Pong game to Hugging Face Spaces!

## ✅ What Was Created

### Core Application Files

1. **`app.py`** (16KB)
   - Main Gradio application
   - Replaces Flask + SocketIO with Gradio interface
   - Includes keyboard controls via JavaScript
   - Automatic HF Hub checkpoint loading
   - Background model loading with progress
   - Real-time frame generation at ~4 FPS

2. **`requirements.txt`** (348 bytes)
   - All Python dependencies for HF Spaces
   - Minimal set: PyTorch, Gradio, HF Hub, numpy, etc.
   - No development dependencies included

### Documentation Files

3. **`HF_README.md`** (5.1KB)
   - This becomes `README.md` on your Space
   - Contains YAML frontmatter for Space metadata
   - Game description and instructions
   - Architecture details
   - Usage guide for visitors

4. **`DEPLOYMENT_GUIDE.md`** (7.4KB)
   - Comprehensive step-by-step deployment guide
   - Two checkpoint options (HF Hub vs Git LFS)
   - Troubleshooting section
   - Cost estimates for different GPU tiers
   - Optimization tips

5. **`QUICK_START.md`** (3.4KB)
   - Condensed "get started in 10 minutes" guide
   - Three deployment options
   - Checklist and common issues
   - Quick reference table

### Utility Scripts

6. **`setup_hf_space.py`** (9.9KB)
   - **Automated deployment script**
   - Interactive CLI wizard
   - Checks dependencies (git, git-lfs, huggingface-cli)
   - Copies files to target directory
   - Sets up Git and Git LFS
   - Commits and pushes to HF
   - Usage: `python setup_hf_space.py`

7. **`upload_checkpoint_to_hf.py`** (4.8KB)
   - **Upload model checkpoint to HF Model Hub**
   - Interactive script with progress display
   - Handles authentication
   - Creates model repo
   - Provides updated config path
   - Usage: `python upload_checkpoint_to_hf.py`

### Configuration Files

8. **`.gitattributes`** (515 bytes)
   - Git LFS configuration
   - Tracks model files (*.pt, *.pth, etc.)
   - Tracks dataset files (*.npy, *.npz)
   - Required if including checkpoint in Space repo

9. **`.gitignore_hf`** (Not for direct use)
   - Template .gitignore for HF Space
   - Copy this to `.gitignore` in your Space repo
   - Excludes development files, caches, logs

## 🎯 Two Deployment Approaches

### Approach A: HF Model Hub (Recommended)

**Pros:**
- Faster checkpoint loading
- Easier to update model
- Checkpoint versioning
- Can use across multiple Spaces

**Steps:**
1. Run `python upload_checkpoint_to_hf.py`
2. Update `configs/inference.yaml` with `hf://` path
3. Deploy Space with `setup_hf_space.py`

**Best for:** Production, public demos, iterating on model

### Approach B: Git LFS in Space

**Pros:**
- Everything in one repo
- No separate model repo needed
- Simpler for small demos

**Steps:**
1. Run `setup_hf_space.py`
2. Choose option 2 when prompted about checkpoint
3. Checkpoint gets included with Git LFS

**Best for:** One-off demos, private Spaces, small models

## 📁 What to Deploy to Your Space

Minimum required files:
```
your-space/
├── app.py                      # Gradio app (REQUIRED)
├── requirements.txt            # Dependencies (REQUIRED)
├── README.md                   # From HF_README.md (REQUIRED)
├── .gitattributes             # If using Git LFS
├── configs/
│   └── inference.yaml         # Model config (REQUIRED)
└── src/                        # Your source code (REQUIRED)
    ├── models/
    ├── nn/
    ├── trainers/
    ├── datasets/
    └── utils/
```

Optional (if using Approach B):
```
└── experiments/
    └── .../
        └── checkpoint.pt       # Model weights with Git LFS
```

## 🚀 Quick Deployment Commands

### Full Automated Deployment
```bash
cd /share/u/wendler/code/toy-wm
python setup_hf_space.py
```

### Just Upload Checkpoint
```bash
cd /share/u/wendler/code/toy-wm
python upload_checkpoint_to_hf.py
```

### Manual Deployment
```bash
# See QUICK_START.md for step-by-step commands
```

## 🔑 Key Differences from Flask Version

| Feature | Flask (`play_pong.py`) | Gradio (`app.py`) |
|---------|------------------------|-------------------|
| Web framework | Flask + SocketIO | Gradio |
| Real-time communication | WebSocket | Gradio events |
| Keyboard input | JavaScript + emit | JavaScript + Gradio |
| Frame streaming | socketio.emit | Queue + periodic update |
| Async mode | eventlet | Built-in Gradio |
| Port | 5000 | 7860 (HF standard) |
| Static files | `static/` folder | Inline in Gradio |
| Model loading | Synchronous | Background thread |

## 📊 File Dependencies

```
app.py
├── requires: configs/inference.yaml
├── imports from: src/
│   ├── src/utils/checkpoint.py
│   ├── src/trainers/diffusion_forcing.py
│   ├── src/datasets/pong1m.py
│   ├── src/config.py
│   └── src/models/dit_dforce.py (indirectly)
└── loads: model checkpoint
    ├── Option A: from HF Hub
    └── Option B: from experiments/
```

## 🎮 How the Gradio App Works

1. **Startup:**
   - Flask server replaced with Gradio app
   - Model loads in background thread (non-blocking)
   - "Ready" status shows when model is loaded

2. **Game Loop:**
   - Background thread generates frames continuously
   - Frames placed in queue (max 2 to prevent lag)
   - Gradio polls queue every 250ms (4 FPS)
   - Displayed in `gr.Image` component

3. **User Input:**
   - JavaScript listens for keyboard events
   - Dispatches custom events to Gradio
   - Updates global `latest_action` variable
   - Next frame generation uses new action

4. **Frame Generation:**
   - Same diffusion model as Flask version
   - Uses KV-caching for efficiency
   - Converts tensor to PIL Image
   - Queued for display

## 🛠️ Customization Options

### Change Frame Rate
In `app.py`, modify:
```python
generator = FrameGenerator(fps=4, ...)  # Change fps here
demo.load(..., every=0.25)               # And here (1/fps)
```

### Change Diffusion Steps
In `app.py`, modify:
```python
generator = FrameGenerator(n_steps=4, ...)  # More steps = better quality, slower
```

### Add Game Features
Ideas to extend the app:
- Score tracking
- Multiple difficulty levels
- Side-by-side real Pong comparison
- Record and download gameplay
- Train on user data

## 📝 Testing Locally Before Deployment

```bash
cd /share/u/wendler/code/toy-wm

# Install Gradio
pip install gradio huggingface-hub

# Run app
python app.py

# Open browser to http://localhost:7860
```

Make sure it works locally before deploying!

## 🔍 Troubleshooting the Gradio App

### Model Not Loading
- Check `configs/inference.yaml` path
- Verify checkpoint exists
- Check console logs for errors

### Keyboard Not Responding
- Ensure page has focus
- Check browser console for JS errors
- Try different browser (Chrome/Firefox best)

### Black Frames
- Wait for model to fully load
- Check GPU availability
- Verify bfloat16 support

### "CUDA not available"
- Only happens in HF Space without GPU
- Select GPU hardware in Space settings

## 🎯 Next Steps

1. **Choose deployment approach** (A or B)
2. **Run appropriate script**:
   - Automated: `python setup_hf_space.py`
   - Upload only: `python upload_checkpoint_to_hf.py`
   - Manual: Follow `QUICK_START.md`
3. **Test your Space** once deployed
4. **Share with the world!** 🌍

## 📚 Additional Resources

- **HF Spaces Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs/
- **HF Hub Docs:** https://huggingface.co/docs/huggingface_hub/
- **Git LFS Docs:** https://git-lfs.github.com/

## 🤝 Contributing

Found an issue or improvement?
- Open an issue on the Space
- Share your modifications
- Help others in HF forums

## 📄 License

All deployment files are MIT licensed - use freely!

---

**Ready to deploy?** Start with `QUICK_START.md` or run `python setup_hf_space.py`! 🚀🎮




