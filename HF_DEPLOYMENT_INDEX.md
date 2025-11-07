# 🎮 Neural Pong → Hugging Face Spaces Deployment

**A complete package to deploy your diffusion-powered Pong game to Hugging Face Spaces**

---

## 🎯 Start Here

**New to deployment?** → Read [`QUICK_START.md`](QUICK_START.md) (3 min read)

**Want full control?** → Read [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) (10 min read)

**Just want it deployed?** → Run `python setup_hf_space.py` (10 min process)

---

## 📚 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **QUICK_START.md** | TL;DR deployment guide | First time deploying |
| **DEPLOYMENT_GUIDE.md** | Comprehensive step-by-step guide | Need detailed instructions |
| **HF_SPACE_FILES_SUMMARY.md** | Overview of all created files | Understanding the package |
| **HF_DEPLOYMENT_INDEX.md** | This file - navigation hub | Finding what you need |

## 🛠️ Utility Scripts

| Script | What It Does | Command |
|--------|--------------|---------|
| **setup_hf_space.py** | Automated full deployment | `python setup_hf_space.py` |
| **upload_checkpoint_to_hf.py** | Upload model to HF Hub | `python upload_checkpoint_to_hf.py` |

## 📦 Application Files (What Gets Deployed)

| File | Size | Purpose |
|------|------|---------|
| **app.py** | 16KB | Main Gradio application |
| **requirements.txt** | 348B | Python dependencies |
| **HF_README.md** | 5.1KB | Space description (becomes README.md) |
| **.gitattributes** | 515B | Git LFS configuration |
| **.gitignore_hf** | - | Template gitignore for Space |

## 🗂️ Existing Files (You Already Have)

These files from your project are needed for deployment:

- `configs/inference.yaml` - Model configuration
- `src/` - Your source code (models, trainers, etc.)
- `experiments/.../checkpoint.pt` - Trained model weights

---

## 🚀 Three Ways to Deploy

### Option 1: Fully Automated (Easiest)

```bash
python setup_hf_space.py
```

**Best for:** First-time users, quick deployment

**What it does:**
- ✅ Checks all dependencies
- ✅ Copies necessary files
- ✅ Sets up Git and Git LFS
- ✅ Commits and pushes to HF
- ✅ Provides next steps

---

### Option 2: Checkpoint on HF Hub (Recommended for Large Models)

```bash
# Step 1: Upload checkpoint
python upload_checkpoint_to_hf.py

# Step 2: Update config to use HF path
# Edit configs/inference.yaml:
#   checkpoint: "hf://YOUR_USERNAME/neural-pong-checkpoint/model.pt"

# Step 3: Deploy Space
python setup_hf_space.py
```

**Best for:** Production, public demos, models >100MB

**Advantages:**
- ⚡ Faster loading times
- 🔄 Easy to update model
- 📦 Reusable across Spaces
- 🎯 Version control for models

---

### Option 3: Manual (Full Control)

Follow step-by-step instructions in [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)

**Best for:** Custom setups, learning the process

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:

- [ ] **Hugging Face account** ([sign up](https://huggingface.co/join))
- [ ] **Git installed** (`git --version`)
- [ ] **Git LFS installed** (`git lfs version`)
- [ ] **Trained model checkpoint** (the `.pt` file)
- [ ] **Config file** (`configs/inference.yaml` with correct checkpoint path)
- [ ] **Source code** (the `src/` directory)

Optional but recommended:
- [ ] **HF CLI installed** (`pip install huggingface-hub`)
- [ ] **Logged into HF** (`huggingface-cli login`)

---

## 🎯 Deployment Decision Tree

```
Start: I want to deploy my Neural Pong game
│
├─ Is my checkpoint > 100MB?
│  │
│  ├─ YES → Use HF Hub approach
│  │         1. Run upload_checkpoint_to_hf.py
│  │         2. Update configs/inference.yaml
│  │         3. Run setup_hf_space.py
│  │
│  └─ NO → Use Git LFS approach
│            1. Run setup_hf_space.py
│            2. Choose option 2 when prompted
│
└─ Do I want full control?
   │
   ├─ YES → Follow DEPLOYMENT_GUIDE.md manually
   │
   └─ NO → Run setup_hf_space.py (automated)
```

---

## 🆘 Troubleshooting

### "Command not found: python"
Try `python3` instead of `python`

### "No module named 'huggingface_hub'"
```bash
pip install huggingface-hub
```

### "git: command not found"
Install Git: https://git-scm.com/downloads

### "git-lfs: command not found"
Install Git LFS: https://git-lfs.github.com/

### "Checkpoint not found"
Check that `configs/inference.yaml` points to the correct path

### Still stuck?
1. Read [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) troubleshooting section
2. Check your terminal output for specific errors
3. Ask in [HF forums](https://discuss.huggingface.co/)

---

## 📊 What to Expect

### Timeline
- **Setup scripts:** 2-3 minutes
- **Uploading files:** 2-5 minutes (depends on checkpoint size)
- **HF Space build:** 5-10 minutes
- **Model loading:** 30-60 seconds (first run)

### Costs (if using paid GPU)
- **T4 small:** ~$0.60/hour (suitable for demos)
- **T4 medium:** ~$1.50/hour (better performance)
- **A10G small:** ~$3.15/hour (production quality)

**Tip:** Enable "Sleep after inactivity" to save costs!

### Performance
- **Expected FPS:** 3-5 on T4, 8-10 on A10G
- **Startup time:** ~60 seconds for model loading
- **Diffusion steps:** 4 (configurable in app.py)

---

## 🎨 Customization After Deployment

Want to modify your deployed Space?

1. Clone your Space repo:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/neural-pong
   ```

2. Make changes to `app.py` or other files

3. Push changes:
   ```bash
   git add .
   git commit -m "Updated: ..."
   git push
   ```

4. HF will automatically rebuild your Space!

**Common customizations:**
- Change FPS (search for `fps=4` in app.py)
- Adjust diffusion steps (search for `n_steps=4`)
- Modify UI colors/layout (Gradio CSS in app.py)
- Add new features (scoring, difficulty levels, etc.)

---

## 📖 Learning Resources

### Hugging Face Spaces
- [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [GPU Spaces](https://huggingface.co/docs/hub/spaces-gpus)

### Gradio
- [Gradio Docs](https://gradio.app/docs/)
- [Gradio Examples](https://gradio.app/demos/)
- [Custom Components](https://gradio.app/custom-components/)

### Git LFS
- [Git LFS Tutorial](https://git-lfs.github.com/)
- [HF Git LFS](https://huggingface.co/docs/hub/repositories-getting-started#git-lfs)

---

## 🎯 Success Criteria

You'll know your deployment succeeded when:

✅ Space URL loads without errors
✅ You see "Ready! Click 'Start Game' to begin."
✅ Clicking "Start Game" shows frames (not black screen)
✅ Keyboard controls (↑/↓ or W/S) change the action display
✅ Frames update in real-time (~4 FPS)

---

## 🌟 After Deployment

### Share Your Space
- Tweet with @huggingface tag
- Share on Reddit/LinkedIn
- Add to your portfolio
- Pin on your HF profile

### Monitor Usage
- Check Space analytics (visitors, likes)
- Review build logs for errors
- Monitor GPU usage/costs

### Iterate
- Update model checkpoint
- Add new features
- Improve UI/UX
- Optimize performance

---

## 📞 Get Help

- **HF Forums:** https://discuss.huggingface.co/
- **Gradio Discord:** https://discord.gg/gradio
- **Documentation Issues:** Open an issue in your Space repo

---

## 🎉 You're Ready!

Pick your path:

- **Just do it:** `python setup_hf_space.py`
- **Learn first:** Read `QUICK_START.md`
- **Understand everything:** Read `DEPLOYMENT_GUIDE.md`

**Good luck with your deployment! 🚀🎮**

---

*Last updated: 2025-11-07*
*Files created for toy-wm Neural Pong project*




