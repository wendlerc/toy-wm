#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch as t

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.datasets.pong1m import fixed2frame
from src.inference.sampling import sample_video
from src.utils.checkpoint import load_action_model_from_config, load_model_from_config
from src.eval.control import annotate_frames


def resolve_experiment_dir(exp_dir: str) -> str:
    if os.path.isdir(exp_dir):
        return exp_dir
    alt = exp_dir.replace("dry-frog-66", "dry-fog-66")
    if alt != exp_dir and os.path.isdir(alt):
        print(f"Experiment dir not found, using '{alt}' instead.")
        return alt
    raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")


def get_device() -> str:
    if t.backends.mps.is_available():
        return "mps"
    if t.cuda.is_available():
        return "cuda"
    return "cpu"


def set_action_embeddings(model: t.nn.Module, action_model: t.nn.Module) -> None:
    with t.no_grad():
        codebook = action_model.learnt_actions
        action_emb = t.cat(
            [
                action_model.unconditional_action.view(1, -1),
                codebook.project_out(codebook.codebook),
            ],
            dim=0,
        )
        model.action_emb.weight.copy_(action_emb)


def write_video(frames_hwc: np.ndarray, path: str, fps: int) -> None:
    if frames_hwc.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames, got {frames_hwc.dtype}")
    height, width = frames_hwc.shape[1], frames_hwc.shape[2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    for frame in frames_hwc:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def frames_to_hwc(frames_t: t.Tensor) -> np.ndarray:
    # frames_t: (T, C, H, W) uint8
    return frames_t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def seed_everything(seed: int) -> None:
    t.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def generate_action_samples(
    model: t.nn.Module,
    action_id: int,
    num_samples: int,
    num_frames: int,
    n_steps: int,
    fps: int,
    out_dir: str,
    annotate: bool,
    base_seed: int,
    batch_size: int,
) -> List[np.ndarray]:
    samples = []
    batch_size = max(1, int(batch_size))
    for start_idx in range(0, num_samples, batch_size):
        current_bs = min(batch_size, num_samples - start_idx)
        seed = base_seed + action_id * num_samples + start_idx
        seed_everything(seed)
        actions = t.full(
            (current_bs, num_frames),
            action_id,
            dtype=t.int64,
            device=model.device,
        )
        pred = sample_video(model, actions, n_steps=n_steps)
        frames = fixed2frame(pred)
        if annotate:
            frames = annotate_frames(frames, actions)
        for offset in range(current_bs):
            sample_idx = start_idx + offset
            frames_hwc = frames_to_hwc(frames[offset])
            sample_path = os.path.join(
                out_dir,
                f"action_{action_id}",
                f"sample_{sample_idx:02d}.mp4",
            )
            write_video(frames_hwc, sample_path, fps)
            samples.append(frames_hwc)
            print(f"Saved {sample_path}")
    return samples


def build_grid_video(
    samples_by_action: Dict[int, List[np.ndarray]],
    action_ids: List[int],
    num_samples: int,
    num_frames: int,
    fps: int,
    out_path: str,
) -> None:
    first_frame = samples_by_action[action_ids[0]][0][0]
    frame_h, frame_w = first_frame.shape[:2]
    header_h = 18
    label_w = 52
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    text_color = (0, 0, 0)
    bg_color = (255, 255, 255)

    grid_frames = []
    for t_idx in range(num_frames):
        row_frames = []
        for action_id in action_ids:
            row_samples = samples_by_action[action_id]
            col_frames = [row_samples[s_idx][t_idx] for s_idx in range(num_samples)]
            row_strip = np.concatenate(col_frames, axis=1)
            label_strip = np.full((frame_h, label_w, 3), bg_color, dtype=np.uint8)
            cv2.putText(
                label_strip,
                f"a{action_id}",
                (4, frame_h // 2),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            row_frames.append(np.concatenate([label_strip, row_strip], axis=1))
        grid_body = np.concatenate(row_frames, axis=0)

        header = np.full((header_h, grid_body.shape[1], 3), bg_color, dtype=np.uint8)
        for s_idx in range(num_samples):
            col_x = label_w + s_idx * frame_w + 4
            cv2.putText(
                header,
                f"v{s_idx}",
                (col_x, header_h - 4),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        grid_frame = np.concatenate([header, grid_body], axis=0)
        grid_frames.append(grid_frame)
    grid_frames = np.stack(grid_frames, axis=0)
    write_video(grid_frames, out_path, fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/genie.yaml")
    parser.add_argument("--experiment_dir", type=str, default="experiments/dry-frog-66")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--min_action", type=int, default=0)
    parser.add_argument("--max_action", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=6)
    parser.add_argument("--no_annotate", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use_saved_models", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    exp_dir = resolve_experiment_dir(args.experiment_dir)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    dtype = t.bfloat16 if cfg.train.dtype == "bf16" else t.float32
    device = get_device()
    print(f"Using device={device}, dtype={dtype}")

    if args.use_saved_models:
        model_ckpt = os.path.join(exp_dir, "model.pt")
        action_ckpt = os.path.join(exp_dir, "action_model.pt")
        model = load_model_from_config(args.config, checkpoint_path=model_ckpt)
        action_model = load_action_model_from_config(args.config, checkpoint_path=action_ckpt)
    else:
        model = load_model_from_config(args.config, checkpoint_path=exp_dir)
        action_model = load_action_model_from_config(args.config, checkpoint_path=exp_dir)

    model = model.to(device).to(dtype).eval()
    action_model = action_model.to(device).to(dtype).eval()
    use_flex = bool(getattr(cfg.model, "use_flex", False))
    should_compile = (args.compile or (use_flex and not args.no_compile))
    if should_compile:
        try:
            model = t.compile(model)
            action_model = t.compile(action_model)
            print("Models compiled with torch.compile for acceleration.")
        except Exception as exc:
            print(f"torch.compile failed, running without compilation: {exc}")
    elif use_flex:
        print("Note: use_flex enabled; consider --compile to avoid unfused attention.")
    set_action_embeddings(model, action_model)

    num_frames = args.seconds * args.fps
    min_action = int(args.min_action)
    max_action = int(args.max_action)
    max_supported = int(model.action_emb.num_embeddings) - 1
    if max_action > max_supported:
        print(f"Requested max_action={max_action} but model supports up to {max_supported}.")
        max_action = max_supported
    action_ids = list(range(min_action, max_action + 1))
    print(f"Generating {len(action_ids)} actions: {action_ids}")

    samples_by_action: Dict[int, List[np.ndarray]] = {}
    for action_id in action_ids:
        samples_by_action[action_id] = generate_action_samples(
            model=model,
            action_id=action_id,
            num_samples=args.num_samples,
            num_frames=num_frames,
            n_steps=args.n_steps,
            fps=args.fps,
            out_dir=results_dir,
            annotate=not args.no_annotate,
            base_seed=args.seed,
            batch_size=args.batch_size,
        )

    grid_path = os.path.join(results_dir, "action_grid.mp4")
    build_grid_video(
        samples_by_action=samples_by_action,
        action_ids=action_ids,
        num_samples=args.num_samples,
        num_frames=num_frames,
        fps=args.fps,
        out_path=grid_path,
    )
    print(f"Saved grid video to {grid_path}")


if __name__ == "__main__":
    t.set_float32_matmul_precision("high")
    main()
