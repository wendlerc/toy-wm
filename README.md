# TLDR

A toy implementation of a diffusion transformer based "world model" trained on 9 hours of pong. Shoutout @pufferlib for their great pong environment that was used for dataset creation.

# Setup

1. install dependencies using `uv sync`
2. download pong dataset `uv run scripts/download_dataset.py`
3. download model `uv run scripts/download_model.py`

# Training

I created my current best checkpoint in a little bit of an ad-hoc way. I first trained a single frame model using: 

`uv run python -m src.main --config configs/bigger_1frame.yaml`

Then, I continued training by updating the `checkpoint` field in `configs/bigger_30frame_causal.yaml` to start from the single-frame model. You can also train a bidirectional one using `configs/bigger_30frame.yaml` but the frame-autoregressive one is much better for the demo app because it supports KV-caching.

`uv run python -m src.main --config configs/bigger_30frame_causal.yaml`

# Inference / running the demo

Update `configs/inference.yaml` to use the checkpoint you want to run. Then run `uv run python play_pong.py`. This should start a server running pong that you can connect to and play interactively. There is also `generate_with_cache.ipynb` to play around with inference.