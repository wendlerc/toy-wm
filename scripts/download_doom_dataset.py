#!/usr/bin/env python3
"""Download Doom latent dataset from Hugging Face Hub.

The dataset contains DC-AE encoded latent frames from Doom PvP gameplay,
stored as WebDataset shards (~4.1 GB each, 188 shards total, ~770 GB).

Usage:
    # Download 5 shards (~20 GB) for a quick start:
    uv run scripts/download_doom_dataset.py

    # Download 1 shard (~4 GB) for a minimal test:
    uv run scripts/download_doom_dataset.py --n-shards 1

    # Download everything (~770 GB):
    uv run scripts/download_doom_dataset.py --all

    # Custom output directory:
    uv run scripts/download_doom_dataset.py --output /path/to/dir
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "chrisxx/doom-2players-latents"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "datasets" / "doom_latents"
DEFAULT_N_SHARDS = 5


def main():
    parser = argparse.ArgumentParser(description="Download Doom latent dataset from HuggingFace")
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS,
                        help=f"Number of shards to download (default: {DEFAULT_N_SHARDS}, each ~4.1 GB)")
    parser.add_argument("--all", action="store_true",
                        help="Download all shards (~770 GB)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # List available shard files
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    shard_files = sorted([f for f in all_files if f.startswith("data/latent-") and f.endswith(".tar")])
    print(f"Available shards: {len(shard_files)} (~{len(shard_files) * 4.1:.0f} GB total)")

    if args.all:
        files_to_download = shard_files
    else:
        files_to_download = shard_files[:args.n_shards]

    print(f"Downloading {len(files_to_download)} shard(s) (~{len(files_to_download) * 4.1:.0f} GB)...")

    for i, filename in enumerate(files_to_download):
        # Files are stored as data/latent-*.tar on HF, download as latent-*.tar locally
        local_name = filename.replace("data/", "")
        local_path = output_dir / local_name
        if local_path.exists():
            print(f"  [{i+1}/{len(files_to_download)}] {local_name} already exists, skipping")
            continue
        print(f"  [{i+1}/{len(files_to_download)}] Downloading {local_name}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download puts it in data/ subfolder, move it up
            downloaded_path = output_dir / filename  # output_dir/data/latent-*.tar
            if downloaded_path.exists() and not local_path.exists():
                downloaded_path.rename(local_path)
            print(f"    -> {local_path}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Clean up empty data/ subdir if it exists
    data_subdir = output_dir / "data"
    if data_subdir.exists() and not any(data_subdir.iterdir()):
        data_subdir.rmdir()

    n_shards = len(list(output_dir.glob("latent-*.tar")))
    print(f"\nDone! {n_shards} shard(s) in {output_dir.absolute()}")
    print(f"To train: uv run python -m src.main --config configs/doom_diffusion_forcing.yaml")


if __name__ == "__main__":
    main()
