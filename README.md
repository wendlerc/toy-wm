# TLDR

A toy implementation of a diffusion transformer based "world model" trained on 9 hours of pong. Shoutout [@pufferlib](https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/ocean/pong/pong.h) for their great pong environment that was used for dataset creation.

The only optimization this codebase really uses so far is `flexattention` but even without it you can train a pong simulator within a reasonable budget.

The folder structure and repo are hopefully self explanatory. If you have any questions please don't hesitate to create an issue.

# Setup

1. install dependencies using `uv sync`
2. download pong dataset `uv run scripts/download_dataset.py`

# Training

You can train your own pong simulator using (should take <= 30 minutes on a A6000):

`uv run python -m src.main`


# Inference / running the demo

Update `configs/inference.yaml` to use the checkpoint you want to run. By default, the checkpoints will be in `./experiments/wandb-run-name`. If you want to play with your model while it is training you can put the run folder into the checkpoint field. Then run `uv run python play_pong.py`. This should start a server running pong that you can connect to and play interactively. There is also `generate_with_cache.ipynb` to play around with inference.

# Resources

Other repositories that I found useful along the way:

- [minRF](https://github.com/cloneofsimo/minRF)
- [owl-wms](https://github.com/Wayfarer-Labs/owl-wms)

