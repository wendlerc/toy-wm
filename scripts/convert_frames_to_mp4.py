#!/usr/bin/env python3
"""
Convert pong frames from .npy file to MP4 video.
"""

import numpy as np
import cv2
import argparse
from pathlib import Path


def convert_frames_to_mp4(
    frames_path: str,
    output_path: str,
    fps: int = 30,
    dataset_name: str = "pong1M",
    max_frames: int = None
):
    """
    Convert frames from .npy file to MP4 video.
    
    Args:
        frames_path: Path to the frames.npy file
        output_path: Path to save the output MP4 file
        fps: Frames per second for the video
        dataset_name: Name of the dataset (pong1M or pong1M_)
        max_frames: Maximum number of frames to process (None for all frames)
    """
    print(f"Loading frames from {frames_path}...")
    frames = np.load(frames_path)
    
    print(f"Frames shape: {frames.shape}")
    print(f"Frames dtype: {frames.dtype}")
    
    # Handle different possible shapes
    if len(frames.shape) == 4:
        # Shape: (num_frames, height, width, channels)
        num_frames, height, width, channels = frames.shape
    elif len(frames.shape) == 5:
        # Shape: (num_episodes, frames_per_episode, height, width, channels)
        # Flatten to (total_frames, height, width, channels)
        num_episodes, frames_per_episode, height, width, channels = frames.shape
        frames = frames.reshape(-1, height, width, channels)
        num_frames = frames.shape[0]
        print(f"Reshaped to: {frames.shape}")
    else:
        raise ValueError(f"Unexpected frames shape: {frames.shape}")
    
    # Ensure frames are uint8 (0-255)
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Limit frames if max_frames is specified
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
        frames = frames[:num_frames]
        print(f"Limited to first {num_frames} frames")
    
    # Ensure BGR format for OpenCV (if RGB, convert to BGR)
    if channels == 3:
        # Assume RGB, convert to BGR for OpenCV
        frames = frames[..., ::-1]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Writing {num_frames} frames to {output_path} at {fps} fps...")
    
    for i, frame in enumerate(frames):
        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_frames} frames...")
    
    out.release()
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pong frames from .npy file to MP4 video"
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
        help="Output MP4 file path (default: datasets/{dataset}/frames.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the video (default: 30)"
    )
    parser.add_argument(
        "--frames-path",
        type=str,
        default=None,
        help="Path to frames.npy file (default: datasets/{dataset}/frames.npy)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in minutes to extract (overrides --max-frames if set)"
    )
    
    args = parser.parse_args()
    
    # Calculate max_frames from duration if provided
    if args.duration is not None:
        args.max_frames = int(args.duration * 60 * args.fps)
        print(f"Extracting {args.duration} minutes = {args.max_frames} frames at {args.fps} fps")
    
    # Set default paths
    if args.frames_path is None:
        frames_path = f"datasets/{args.dataset}/frames.npy"
    else:
        frames_path = args.frames_path
    
    if args.output is None:
        output_path = f"datasets/{args.dataset}/frames.mp4"
    else:
        output_path = args.output
    
    # Check if frames file exists
    if not Path(frames_path).exists():
        raise FileNotFoundError(f"Frames file not found: {frames_path}")
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    convert_frames_to_mp4(
        frames_path=frames_path,
        output_path=output_path,
        fps=args.fps,
        dataset_name=args.dataset,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

