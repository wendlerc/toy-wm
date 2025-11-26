#!/usr/bin/env python3
"""
Mirror pong frames across the vertical axis (horizontal flip).
"""

import numpy as np
import argparse
from pathlib import Path


def mirror_frames(
    frames_path: str,
    output_path: str,
    dataset_name: str = "pong1M"
):
    """
    Mirror frames horizontally and save to a new .npy file.
    
    Args:
        frames_path: Path to the frames.npy file
        output_path: Path to save the mirrored frames
        dataset_name: Name of the dataset (pong1M or pong1M_)
    """
    print(f"Loading frames from {frames_path}...")
    frames = np.load(frames_path)
    
    original_shape = frames.shape
    print(f"Original frames shape: {original_shape}")
    print(f"Frames dtype: {frames.dtype}")
    
    # Handle different possible shapes
    if len(frames.shape) == 4:
        # Shape: (num_frames, height, width, channels)
        num_frames, height, width, channels = frames.shape
        # Flip horizontally (mirror across vertical axis)
        # Use [:, :, ::-1, :] to flip the width dimension
        mirrored_frames = frames[:, :, ::-1, :]
    elif len(frames.shape) == 5:
        # Shape: (num_episodes, frames_per_episode, height, width, channels)
        num_episodes, frames_per_episode, height, width, channels = frames.shape
        # Flip horizontally (mirror across vertical axis)
        # Use [:, :, :, ::-1, :] to flip the width dimension
        mirrored_frames = frames[:, :, :, ::-1, :]
    else:
        raise ValueError(f"Unexpected frames shape: {frames.shape}")
    
    print(f"Mirrored frames shape: {mirrored_frames.shape}")
    
    # Save the mirrored frames
    print(f"Saving mirrored frames to {output_path}...")
    np.save(output_path, mirrored_frames)
    
    print(f"Successfully saved {mirrored_frames.shape[0] if len(mirrored_frames.shape) == 4 else mirrored_frames.shape[0] * mirrored_frames.shape[1]} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mirror pong frames across the vertical axis (horizontal flip)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pong1M",
        choices=["pong1M", "pong1M_"],
        help="Dataset name (default: pong1M)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npy file path (default: datasets/{dataset}/frames_mirrored.npy)"
    )
    parser.add_argument(
        "--frames-path",
        type=str,
        default=None,
        help="Path to frames.npy file (default: datasets/{dataset}/frames.npy)"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.frames_path is None:
        frames_path = f"datasets/{args.dataset}/frames.npy"
    else:
        frames_path = args.frames_path
    
    if args.output is None:
        output_path = f"datasets/{args.dataset}/frames_mirrored.npy"
    else:
        output_path = args.output
    
    # Check if frames file exists
    if not Path(frames_path).exists():
        raise FileNotFoundError(f"Frames file not found: {frames_path}")
    
    # Check if output file already exists
    if Path(output_path).exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Please specify a different output path or delete the existing file."
        )
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mirror_frames(
        frames_path=frames_path,
        output_path=output_path,
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()


