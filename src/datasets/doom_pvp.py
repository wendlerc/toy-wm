"""Doom PvP latent dataset loader for world model training.

Loads DC-AE latent clips from WebDataset shards (pre-encoded at 32x spatial
compression, 32 channels, float16). Each PvP episode has two player
perspectives which are treated as independent training sequences.

Latent shape per frame: (32, 15, 20) — this replaces the (3, H, W) RGB frames
used in pong1m. The model operates directly in latent space.

Action space: 14-dimensional vector per frame (13 binary buttons + 1 continuous
turn delta). Actions are projected via nn.Linear instead of nn.Embedding.

Usage:
    loader, pred2frame = get_loader(batch_size=32, fps=35, duration=2)
    for latents, actions in loader:
        # latents: (B, T, 32, 15, 20) float32
        # actions: (B, T, 14) float32
        ...
"""

import io
import math
import random
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch as t
import webdataset as wds
from torch.utils.data import IterableDataset, get_worker_info
from webdataset.filters import _shuffle


# --------------------------------------------------------------------------- #
# pred2frame: decode latents back to RGB for visualization / WandB logging
# --------------------------------------------------------------------------- #

_vae_cache = {}

def _get_vae(device="cuda"):
    """Lazy-load DC-AE VAE decoder (cached)."""
    if device not in _vae_cache:
        from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
        import glob
        local = glob.glob("/tmp/dc-ae-cache/snapshots/*/")
        model_path = local[0] if local else "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
        vae = AutoencoderDC.from_pretrained(
            model_path, torch_dtype=t.float16,
        ).to(device).eval()
        _vae_cache[device] = vae
    return _vae_cache[device]

def offload_vae(device="cuda"):
    """Move cached VAE off the specified device to CPU to free VRAM."""
    vae = _vae_cache.get(device)
    if vae is None:
        return
    vae.to("cpu")
    _vae_cache["cpu"] = vae
    if device != "cpu":
        _vae_cache.pop(device, None)
    if t.cuda.is_available():
        t.cuda.empty_cache()


def latent2frame(z):
    """Convert latent predictions to uint8 RGB frames for visualization.

    Args:
        z: (B, T, 32, 15, 20) or (T, 32, 15, 20) latent tensor

    Returns:
        (B, T, 3, 480, 640) or (T, 3, 480, 640) uint8 tensor
    """
    squeeze = False
    if z.dim() == 4:
        z = z.unsqueeze(0)
        squeeze = True
    B, T, C, H, W = z.shape
    device = z.device
    vae = _get_vae(device)
    frames = []
    with t.no_grad():
        for i in range(0, B * T, 16):
            flat = z.reshape(B * T, C, H, W)[i:i+16].to(t.float16)
            if flat.shape[-2] == 16: # this just comes from our hack to enable 2x2 patches (so we added bunch of zeros...)
                flat = flat[:,:, 1:,:]
            rgb = vae.decode(flat).sample
            rgb = ((rgb.clamp(-1, 1) + 1) / 2 * 255).byte()
            frames.append(rgb)
    frames = t.cat(frames, dim=0)[:B*T].reshape(B, T, 3, 480, 640)
    if squeeze:
        frames = frames.squeeze(0)
    return frames


# --------------------------------------------------------------------------- #
# WebDataset streaming loader (inlined from doom_arena/latent_loader.py)
# --------------------------------------------------------------------------- #

class _SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)
    def set_value(self, epoch):
        self.shared_epoch.value = epoch
    def get_value(self):
        return self.shared_epoch.value


class _DetShuffle(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, _SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            worker_info = get_worker_info()
            seed = worker_info.seed if worker_info else 0
            seed += epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class _ResampledShards(IterableDataset):
    def __init__(self, urls, epoch=-1, deterministic=True):
        super().__init__()
        self.urls = list(urls)
        self.epoch = epoch
        self.deterministic = deterministic
        self.rng = random.Random()

    def __iter__(self):
        if isinstance(self.epoch, _SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            worker_info = get_worker_info()
            seed = (worker_info.seed if worker_info else 0) + epoch
            self.rng.seed(seed)
        while True:
            yield dict(url=self.rng.choice(self.urls))


class _ExplodeClips(wds.PipelineStage):
    def __init__(self, clip_len: int, rng: random.Random, pad_height: bool = True):
        self.clip_len = clip_len
        self.rng = rng
        self.pad_height = pad_height

    def run(self, src):
        for sample in src:
            raw_p1 = sample["latents_p1.npy"]
            n_frames = raw_p1.shape[0]
            if self.pad_height:
                latents_p1 = np.zeros((n_frames, 32, 16, 20), dtype=np.float32)
                latents_p1[:,:, 1:,:] = raw_p1
            else:
                latents_p1 = raw_p1.astype(np.float32)
            n_frames = latents_p1.shape[0]
            n_clips = n_frames // self.clip_len
            if n_clips == 0:
                continue

            actions_p1 = np.zeros((n_frames, 15), dtype=np.float32)
            actions_p2 = np.zeros((n_frames, 15), dtype=np.float32)
            # prepend unconditional action...
            actions_p1[:, 1:] = sample.get("actions_p1.npy", np.zeros((n_frames, 14), dtype=np.float32))

            rewards_p1 = sample.get("rewards_p1.npy", np.zeros(n_frames, dtype=np.float32))
            has_p2 = "latents_p2.npy" in sample
            if has_p2:
                raw_p2 = sample["latents_p2.npy"]
                if self.pad_height:
                    latents_p2 = np.zeros((n_frames, 32, 16, 20), dtype=np.float32)
                    latents_p2[:,:, 1:,:] = raw_p2
                else:
                    latents_p2 = raw_p2.astype(np.float32)
                actions_p2[:, 1:] = sample.get("actions_p2.npy", np.zeros((n_frames, 14), dtype=np.float32))
                rewards_p2 = sample.get("rewards_p2.npy", np.zeros(n_frames, dtype=np.float32))

            starts = list(range(0, n_clips * self.clip_len, self.clip_len))
            self.rng.shuffle(starts)

            for start in starts:
                end = start + self.clip_len
                clip = {
                    "latents_p1": latents_p1[start:end],
                    "actions_p1": actions_p1[start:end],
                    "rewards_p1": rewards_p1[start:end],
                }
                if has_p2:
                    clip["latents_p2"] = latents_p2[start:end]
                    clip["actions_p2"] = actions_p2[start:end]
                    clip["rewards_p2"] = rewards_p2[start:end]
                else:
                    clip["latents_p2"] = np.zeros_like(clip["latents_p1"])
                    clip["actions_p2"] = np.zeros_like(clip["actions_p1"])
                    clip["rewards_p2"] = np.zeros_like(clip["rewards_p1"])
                yield clip


def _collate_clips(batch: list[dict]) -> dict:
    return {
        k: t.from_numpy(np.stack([b[k] for b in batch]))
        for k in ("latents_p1", "latents_p2", "actions_p1", "actions_p2", "rewards_p1", "rewards_p2")
    }


def _decode_all_npy(sample: dict) -> dict:
    for key in list(sample.keys()):
        if key.endswith(".npy") and isinstance(sample[key], bytes):
            sample[key] = np.load(io.BytesIO(sample[key]))
    return sample


def _log_and_continue(exn):
    import logging
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


class _LatentTrainLoader:
    """WebDataset streaming loader for latent shards."""

    def __init__(self, root, clip_len=16, batch_size=32, num_workers=4,
                 seed=42, epoch=0, resampled=True, shard_urls=None, pad_height=True):
        self.root = Path(root)
        self.clip_len = clip_len
        self.batch_size = batch_size
        self._shared_epoch = _SharedEpoch(epoch)

        if shard_urls is None:
            shard_paths = sorted(self.root.glob("latent-*.tar"))
            shard_urls = [str(p) for p in shard_paths]
        assert len(shard_urls) > 0, f"No latent-*.tar shards found in {root}"

        clip_rng = random.Random(seed)

        if resampled:
            pipeline = [_ResampledShards(shard_urls, epoch=self._shared_epoch, deterministic=True)]
        else:
            pipeline = [wds.SimpleShardList(shard_urls)]

        if not resampled:
            pipeline.extend([
                _DetShuffle(bufsize=100, initial=10, seed=seed, epoch=self._shared_epoch),
                wds.split_by_node,
                wds.split_by_worker,
            ])

        pipeline.extend([
            wds.tarfile_to_samples(handler=_log_and_continue),
            wds.map(_decode_all_npy, handler=_log_and_continue),
            _ExplodeClips(clip_len, clip_rng, pad_height=pad_height),
            wds.batched(batch_size, partial=False, collation_fn=_collate_clips),
        ])

        dataset = wds.DataPipeline(*pipeline)

        if resampled:
            num_samples = len(shard_urls) * 13 * (8000 // clip_len)
            global_batch_size = batch_size
            num_batches = math.ceil(num_samples / global_batch_size)
            num_workers_actual = max(1, num_workers)
            num_worker_batches = math.ceil(num_batches / num_workers_actual)
            num_batches = num_worker_batches * num_workers_actual
            dataset = dataset.with_epoch(num_worker_batches)
            self.num_batches = num_batches
        else:
            self.num_batches = None

        self._dataloader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=num_workers, persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    def set_epoch(self, epoch: int):
        self._shared_epoch.set_value(epoch)

    def __iter__(self):
        return iter(self._dataloader)


# --------------------------------------------------------------------------- #
# Wrapper iterator: adapts loader to yield (latents, actions) tuples
# --------------------------------------------------------------------------- #

class _DoomLoaderWrapper:
    """Wraps _LatentTrainLoader to yield (latents, actions) tuples.

    Both player perspectives are interleaved as independent samples,
    effectively doubling the batch size.
    """

    def __init__(self, inner_loader, fps, duration):
        self.inner = inner_loader
        self.clip_len = fps * duration

    def __iter__(self):
        for batch in self.inner:
            for player in ("p1", "p2"):
                latents = batch[f"latents_{player}"]  # (B, T+1, 32, 15, 20)
                actions = batch[f"actions_{player}"]   # (B, T+1, 14)

                # Same frame-action shift as pong1m:
                # each frame is conditioned on the PREVIOUS frame's action
                frames = latents[:, 1:]     # (B, T, 32, 15, 20)
                acts = actions[:, :-1]      # (B, T, 14)

                yield frames.float(), acts.float()


# --------------------------------------------------------------------------- #
# Main entry point (matches pong1m.get_loader interface)
# --------------------------------------------------------------------------- #

def split_shards(root="./datasets/doom_latents", val_fraction=0.1, seed=42):
    """Partition shard URLs into train/val sets.

    Returns:
        (train_urls, val_urls) — disjoint sorted lists of shard paths
    """
    shard_paths = sorted(Path(root).glob("latent-*.tar"))
    shard_urls = [str(p) for p in shard_paths]
    assert len(shard_urls) > 0, f"No latent-*.tar shards found in {root}"
    # Shuffle deterministically, take last val_fraction as val
    shuffled = list(shard_urls)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_urls = sorted(shuffled[:n_val])
    train_urls = sorted(shuffled[n_val:])
    print(f"Shard split: {len(train_urls)} train, {len(val_urls)} val")
    return train_urls, val_urls


def get_loader(batch_size=32, fps=35, duration=2, shuffle=True,
               debug=False, drop_duration=False,
               shard_dir="./datasets/doom_latents", num_workers=8,
               shard_urls=None, pad_height=True, override_clip_len=None):
    """Load Doom PvP latent dataset.

    Args:
        shard_urls: If provided, use these shard URLs instead of globbing shard_dir.
        override_clip_len: If set, use this as the output clip length (frames yielded
            per clip) instead of fps*duration. The internal clip includes +1 for the
            context frame shift.

    Returns:
        (loader, pred2frame) — loader yields (latents, actions) tuples
            latents: (B, T, 32, 15, 20) float32
            actions: (B, T, 14) float32
        pred2frame converts latents to (B, T, 3, 480, 640) uint8 via DC-AE
    """
    if override_clip_len is not None:
        clip_len = override_clip_len + 1  # +1 for the context frame shift
    else:
        clip_len = fps * duration + 1  # +1 for the context frame shift

    inner = _LatentTrainLoader(
        shard_dir,
        clip_len=clip_len,
        batch_size=batch_size,
        num_workers=num_workers,
        resampled=True,
        shard_urls=shard_urls,
        pad_height=pad_height,
    )

    loader = _DoomLoaderWrapper(inner, fps, duration)
    n_shards = len(shard_urls) if shard_urls else "all"
    print(f"Doom PvP loader: clip_len={fps*duration} frames ({duration}s @ {fps}fps), "
          f"batch={batch_size} (x2 for P1+P2), shards={n_shards}, pad_height={pad_height}")

    return loader, latent2frame
