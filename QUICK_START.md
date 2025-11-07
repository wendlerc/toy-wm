# 🚀 Quick Start: Deploy Neural Pong to Hugging Face Spaces

## TL;DR - Get Your Game Online in 10 Minutes

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python setup_hf_space.py
```

Follow the prompts and you're done! 🎉

### Option 2: Manual Setup

```bash
# 1. Upload checkpoint to HF Hub (faster loading)
python upload_checkpoint_to_hf.py

# 2. Update configs/inference.yaml with HF path
# Change: checkpoint: "experiments/.../model.pt"
# To: checkpoint: "hf://YOUR_USERNAME/neural-pong-checkpoint/model.pt"

# 3. Create Space on HF and clone it
huggingface-cli login
git clone https://huggingface.co/spaces/YOUR_USERNAME/neural-pong
cd neural-pong

# 4. Copy files
cp ../toy-wm/app.py .
cp ../toy-wm/requirements.txt .
cp ../toy-wm/HF_README.md README.md
cp ../toy-wm/.gitattributes .
cp -r ../toy-wm/src .
cp -r ../toy-wm/configs .

# 5. Push to HF
git add .
git commit -m "Initial commit"
git push
```

### Option 3: Manual HF Website

1. Go to https://huggingface.co/new-space
2. Create Gradio Space with GPU
3. Upload files via web interface
4. Wait for build

## 📋 Checklist

Before deploying, make sure you have:

- [ ] Trained model checkpoint (.pt file)
- [ ] Hugging Face account
- [ ] Git and Git LFS installed
- [ ] configs/inference.yaml configured

## 🎯 What Gets Deployed

Files you need to upload:
```
neural-pong/
├── app.py                    # Main Gradio app
├── requirements.txt          # Python dependencies
├── README.md                 # Space description (from HF_README.md)
├── .gitattributes           # Git LFS config
├── configs/
│   └── inference.yaml       # Model config
└── src/
    ├── models/              # Model architecture
    ├── nn/                  # Neural network components
    ├── trainers/            # Training/inference logic
    ├── datasets/            # Data loading
    └── utils/               # Utilities
```

Optional (if not using HF Hub for checkpoint):
```
└── experiments/
    └── .../
        └── checkpoint.pt    # Model weights (tracked with Git LFS)
```

## ⚙️ Hardware Selection

| GPU Tier | Cost/Hour | Performance | Use Case |
|----------|-----------|-------------|----------|
| T4 small | $0.60 | ~3-4 FPS | Testing |
| T4 medium | $1.50 | ~4-5 FPS | Demo |
| A10G small | $3.15 | ~8-10 FPS | Production |

Recommendation: **T4 small** for public demos

## 🔧 Common Issues

### "CUDA not available"
→ Select GPU hardware in Space settings

### "Checkpoint not found"
→ Upload checkpoint to HF Hub or use Git LFS

### Slow startup
→ Use HF Hub for checkpoint (faster than Git LFS)

### Keyboard not working
→ Click on game area for focus

## 🎮 Test Your Deployment

Once deployed:

1. Visit your Space URL
2. Wait for "Ready! Click 'Start Game' to begin."
3. Click "Start Game"
4. Press ↑/↓ or W/S to play

## 📖 Full Documentation

- `HF_README.md` - Space description and info
- `DEPLOYMENT_GUIDE.md` - Detailed deployment guide
- `setup_hf_space.py` - Automated setup script
- `upload_checkpoint_to_hf.py` - Checkpoint upload utility

## 🆘 Need Help?

1. Check `DEPLOYMENT_GUIDE.md` for detailed instructions
2. See HF Spaces docs: https://huggingface.co/docs/hub/spaces
3. Ask in HF forums: https://discuss.huggingface.co/

---

Happy deploying! 🚀🎮




