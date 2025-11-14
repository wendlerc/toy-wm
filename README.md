# TLDR

A toy implementation of a diffusion transformer based "world model" trained on 9 hours of pong. Shoutout [@pufferlib](https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/ocean/pong/pong.h) for their great pong environment that was used for dataset creation.

# Setup

1. install dependencies using `uv sync`
2. download pong dataset `uv run scripts/download_dataset.py`
3. download model `uv run scripts/download_model.py`

# Training

I created my current best checkpoint in a little bit of an ad-hoc way on a single A6000. I first trained a single frame model using (starting from a single-frame model accelerates the mult-frame training a lot). 

`uv run python -m src.main --config configs/bigger_1frame.yaml`

Then, I continued training on sequences of 30 frames each with bidirectional attention. This is done by updating the `checkpoint` field in `configs/bigger_30frame.yaml` and running:

`uv run python -m src.main --config configs/bigger_30frame.yaml`

Finally, I started from the bidirectional 30 frame model to create an autoregressive on using diffusion forcing. Again update the `checkpoint` field in `configs/bigger_30frame_causal.yaml` to start from the bidirectional model. The frame autoregressive models are great for the demo app because it supports KV-caching.

`uv run python -m src.main --config configs/bigger_30frame_causal.yaml`

# Inference / running the demo

Update `configs/inference.yaml` to use the checkpoint you want to run. Then run `uv run python play_pong.py`. This should start a server running pong that you can connect to and play interactively. There is also `generate_with_cache.ipynb` to play around with inference.

# Resources

- [minRF](https://github.com/cloneofsimo/minRF)
- [owl-wms](https://github.com/Wayfarer-Labs/owl-wms)

