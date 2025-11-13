#!/usr/bin/env python3
"""
Benchmarking script for t.compile with the pong model.
Generates 180 frames with action 1 and measures performance after burn-in.
Tests different configurations: eager, inference_mode, compile modes, etc.
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch as t
import torch._dynamo as _dynamo
import numpy as np
from PIL import Image

# Project imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.checkpoint import load_model_from_config
from src.trainers.diffusion_forcing import sample
from src.datasets.pong1m import get_loader, fixed2frame
from src.config import Config


def _ensure_cuda():
    if not t.cuda.is_available():
        raise RuntimeError("CUDA GPU required; torch.cuda.is_available() is False.")
    return t.device("cuda:0")


def _reset_cache_fresh(model):
    model.cache.reset()


def _step_eager(model_, action_scalar_long: int, n_steps: int, cfg: float, clamp: bool, device):
    """Step function with dynamo disabled (original behavior)."""
    noise = t.randn(1, 1, 3, 24, 24, device=device)
    action_buf = t.empty((1, 1), dtype=t.long, device=device)
    action_buf.fill_(int(action_scalar_long))

    # Sample with the fresh noise
    z = sample(model_, noise, action_buf, num_steps=n_steps, cfg=cfg, negative_actions=None)
    
    # Update cache location after sample
    model_.cache.update_global_location(1)
    
    if clamp:
        z = t.clamp(z, -1, 1)
    return z

def _step_compilable(model_, action_scalar_long: int, n_steps: int, cfg: float, clamp: bool, device):
    """
    Step function that can be compiled by torch.compile.
    Removed @_dynamo.disable to allow optimization.
    """
    noise = t.randn(1, 1, 3, 24, 24, device=device)
    action_buf = t.empty((1, 1), dtype=t.long, device=device)
    action_buf.fill_(int(action_scalar_long))

    # Sample with the fresh noise
    z = sample(model_, noise, action_buf, num_steps=n_steps, cfg=cfg, negative_actions=None)
    
    # Update cache location after sample
    model_.cache.update_global_location(1)
    
    if clamp:
        z = t.clamp(z, -1, 1)
    return z


def generate_frames(model, pred2frame, actions, step_func, n_steps=4, cfg=0.0, clamp=True, 
                   device=None, burn_in_frames=10, use_inference_mode=True, verbose=True):
    """
    Generate frames for given actions and measure timing after burn-in.
    
    Args:
        model: The model to use
        pred2frame: Function to convert predictions to frames
        actions: Tensor of actions (1, n_frames)
        n_steps: Number of diffusion steps
        cfg: CFG scale
        clamp: Whether to clamp outputs
        device: Device to run on
        burn_in_frames: Number of frames to skip for timing
        use_inference_mode: Whether to use torch.inference_mode()
        verbose: Print progress
    
    Returns:
        frames: Generated frames as numpy array (n_frames, H, W, C)
        timings: List of frame generation times (after burn-in)
    """
    if device is None:
        device = next(model.parameters()).device
    
    _reset_cache_fresh(model)
    
    frames_list = []
    timings = []
    
    # Use inference_mode or no_grad based on configuration
    if use_inference_mode:
        context_mgr = t.inference_mode()
    else:
        context_mgr = t.no_grad()
    
    with context_mgr, t.autocast(device_type="cuda", dtype=t.bfloat16):
        for frame_idx in range(actions.shape[1]):
            action = int(actions[0, frame_idx].item())
            
            # Time the step
            step_start = time.perf_counter()
            z = step_func(model, action_scalar_long=action, n_steps=n_steps, cfg=cfg, clamp=clamp, device=device)
            step_end = time.perf_counter()
            
            # Convert to frame
            frames_btchw = pred2frame(z)
            frame_arr = frames_btchw[0, 0].permute(1, 2, 0).contiguous()
            if isinstance(frame_arr, t.Tensor):
                frame_np = frame_arr.to("cpu", non_blocking=True).numpy()
            else:
                frame_np = frame_arr.astype(np.uint8, copy=False)
            
            frames_list.append(frame_np)
            
            # Only record timings after burn-in
            if frame_idx >= burn_in_frames:
                timings.append(step_end - step_start)
            
            if verbose and (frame_idx + 1) % 20 == 0:
                print(f"Generated {frame_idx + 1}/{actions.shape[1]} frames")
    
    frames = np.stack(frames_list, axis=0)
    return frames, timings


def save_video(frames, output_path, fps=30):
    """Save frames as a GIF or video file."""
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == '.gif':
        # Save as GIF
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"Saved GIF to {output_path}")
    else:
        # Save as individual frames
        output_dir = output_path.parent / output_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(output_dir / f"frame_{i:04d}.png")
        print(f"Saved {len(frames)} frames to {output_dir}")


def benchmark_model(use_compile=False, compile_mode="default", use_inference_mode=True,
                   compile_step=True, n_frames=500, n_steps=4, cfg=0.0, 
                   clamp=True, burn_in_frames=10, output_path=None, verbose=True):
    """
    Benchmark the model with or without torch.compile.
    
    Args:
        use_compile: Whether to use t.compile
        n_frames: Number of frames to generate
        n_steps: Number of diffusion steps per frame
        cfg: CFG scale
        clamp: Whether to clamp outputs
        burn_in_frames: Number of frames to skip for timing
        output_path: Path to save output video (optional)
        verbose: Print detailed information
    """
    print("=" * 80)
    print(f"Benchmarking model {'WITH' if use_compile else 'WITHOUT'} torch.compile")
    print("=" * 80)
    
    # Setup
    device = _ensure_cuda()
    print(f"Using device: {device}")
    
    # Performance settings
    t.backends.cudnn.benchmark = True
    t.backends.cudnn.conv.fp32_precision = "tf32"
    t.backends.cuda.matmul.fp32_precision = "high"
    
    # Load config
    config_path = os.path.join(project_root, "configs/inference.yaml")
    config = Config.from_yaml(config_path)
    checkpoint_path = config.model.checkpoint
    
    # Load model
    print("\nLoading model...")
    load_start = time.time()
    model = load_model_from_config(config_path, checkpoint_path=checkpoint_path, strict=False)
    model.to(device)
    model.eval()
    model.activate_caching(1)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Choose step function based on compilation settings
    if compile_step and use_compile:
        step_func = _step_compilable
        print(f"\nUsing compilable step function (will be optimized by torch.compile)")
    else:
        step_func = _step_eager
        if use_compile:
            print(f"\nUsing eager step function (@_dynamo.disable) - step function won't be compiled")
    
    # Compile model if requested
    compile_time = 0
    step_compile_time = 0
    if use_compile:
        print(f"\nCompiling model with torch.compile (mode={compile_mode})...")
        print(f"Model type before compile: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Fix recompilation issue: allow dynamic integer attributes on nn.Module
        # This prevents recompilation when cache.local_loc changes
        print("\nConfiguring torch._dynamo to handle dynamic cache attributes...")
        try:
            t._dynamo.config.allow_unspec_int_on_nn_module = True
            print("✓ Enabled allow_unspec_int_on_nn_module (prevents recompilation from cache.local_loc changes)")
        except Exception as e:
            print(f"⚠ Could not set allow_unspec_int_on_nn_module: {e}")
        
        compile_start = time.time()
        try:
            if compile_mode == "default":
                model = t.compile(model)
            elif compile_mode == "reduce-overhead":
                model = t.compile(model, mode="reduce-overhead")
            elif compile_mode == "max-autotune":
                model = t.compile(model, mode="max-autotune")
            elif compile_mode == "max-autotune-no-cudagraphs":
                model = t.compile(model, mode="max-autotune-no-cudagraphs")
            else:
                model = t.compile(model, mode=compile_mode)
            compile_time = time.time() - compile_start
            print(f"Model compiled in {compile_time:.2f}s")
            print(f"Model type after compile: {type(model)}")
            
            # Check if compilation actually happened
            if hasattr(model, '_orig_mod'):
                print("✓ Compilation wrapper detected (_orig_mod exists)")
                print(f"  Original model type: {type(model._orig_mod)}")
            else:
                print("⚠ Warning: No _orig_mod found - compilation may not have worked")
            
            # Compile step function if requested
            if compile_step:
                print(f"\nCompiling step function with torch.compile (mode={compile_mode})...")
                step_compile_start = time.time()
                try:
                    if compile_mode == "default":
                        step_func = t.compile(step_func)
                    elif compile_mode == "reduce-overhead":
                        step_func = t.compile(step_func, mode="reduce-overhead")
                    elif compile_mode == "max-autotune":
                        step_func = t.compile(step_func, mode="max-autotune")
                    elif compile_mode == "max-autotune-no-cudagraphs":
                        step_func = t.compile(step_func, mode="max-autotune-no-cudagraphs")
                    else:
                        step_func = t.compile(step_func, mode=compile_mode)
                    step_compile_time = time.time() - step_compile_start
                    print(f"Step function compiled in {step_compile_time:.2f}s")
                    print(f"Step function type after compile: {type(step_func)}")
                except Exception as e:
                    print(f"⚠ ERROR compiling step function: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with uncompiled step function...")
                    step_func = _step_eager
            
            # Try to get compilation stats if available
            try:
                if hasattr(t._dynamo, 'config'):
                    print(f"  Dynamo cache size: {getattr(t._dynamo.config, 'cache_size', 'unknown')}")
            except:
                pass
                
        except Exception as e:
            print(f"⚠ ERROR during compilation: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with uncompiled model...")
            use_compile = False
    
    # Load dataset utilities
    _, pred2frame = get_loader(duration=1, fps=30, mode='-1,1')
    
    # Prepare actions: n_frames frames with action 1
    actions = t.tensor([[1] * n_frames], dtype=t.long, device=device)
    print(f"\nGenerating {n_frames} frames with action 1...")
    
    # Warmup (burn-in)
    print(f"\nRunning {burn_in_frames} warmup frames...")
    _reset_cache_fresh(model)
    warmup_context = t.inference_mode() if use_inference_mode else t.no_grad()
    
    warmup_times = []
    with warmup_context, t.autocast(device_type="cuda", dtype=t.bfloat16):
        for i in range(burn_in_frames):
            warmup_start = time.perf_counter()
            _ = step_func(model, action_scalar_long=1, n_steps=n_steps, cfg=cfg, clamp=clamp, device=device)
            warmup_end = time.perf_counter()
            warmup_times.append(warmup_end - warmup_start)
            
            if verbose:
                if i < 5 or (i + 1) % 10 == 0:
                    print(f"  Warmup {i + 1}/{burn_in_frames}: {warmup_times[-1]*1000:.2f}ms")
    
    if warmup_times:
        avg_warmup = np.mean(warmup_times)
        print(f"  Warmup complete - avg: {avg_warmup*1000:.2f}ms, first: {warmup_times[0]*1000:.2f}ms, last: {warmup_times[-1]*1000:.2f}ms")
        if use_compile and len(warmup_times) >= 3:
            # Check if first few frames are slower (compilation happening)
            first_3_avg = np.mean(warmup_times[:3])
            last_3_avg = np.mean(warmup_times[-3:])
            if first_3_avg > last_3_avg * 1.5:
                print(f"  ⚠ First frames slower ({first_3_avg*1000:.2f}ms) than last ({last_3_avg*1000:.2f}ms) - compilation may still be happening")
    
    # Synchronize GPU before timing
    t.cuda.synchronize()
    
    # Generate frames and measure timing
    print(f"\nGenerating {n_frames} frames (timing after {burn_in_frames} burn-in frames)...")
    print(f"Note: First {burn_in_frames} frames are warmup and excluded from timing")
    
    # Enable detailed logging for compile (optional, can be verbose)
    if use_compile and verbose and False:  # Disabled by default - set to True for debugging
        import logging
        # Set torch._dynamo logging if available
        try:
            t._dynamo.config.verbose = True
            t._dynamo.config.log_level = logging.INFO
        except:
            pass
    
    gen_start = time.time()
    frames, timings = generate_frames(
        model, pred2frame, actions, step_func,
        n_steps=n_steps, cfg=cfg, clamp=clamp,
        device=device, burn_in_frames=burn_in_frames, 
        use_inference_mode=use_inference_mode, verbose=verbose
    )
    t.cuda.synchronize()
    gen_end = time.time()
    
    # Log first few measured frame times to check for compilation overhead
    if timings and verbose:
        print(f"\nFirst 5 measured frame times: {[f'{t*1000:.2f}ms' for t in timings[:5]]}")
        print(f"Last 5 measured frame times: {[f'{t*1000:.2f}ms' for t in timings[-5:]]}")
    
    # Calculate statistics
    total_time = gen_end - gen_start
    measured_frames = len(timings)
    avg_frame_time = np.mean(timings) if timings else 0
    std_frame_time = np.std(timings) if timings else 0
    min_frame_time = np.min(timings) if timings else 0
    max_frame_time = np.max(timings) if timings else 0
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - torch.compile: {use_compile}")
    if use_compile:
        print(f"  - Compile mode: {compile_mode}")
        print(f"  - Model compile time: {compile_time:.2f}s")
        print(f"  - Compile step function: {compile_step}")
        if compile_step:
            print(f"  - Step compile time: {step_compile_time:.2f}s")
    print(f"  - Inference mode: {use_inference_mode}")
    print(f"  - Frames generated: {n_frames}")
    print(f"  - Burn-in frames: {burn_in_frames}")
    print(f"  - Measured frames: {measured_frames}")
    print(f"  - Diffusion steps per frame: {n_steps}")
    print(f"  - CFG scale: {cfg}")
    print(f"\nTiming (after burn-in):")
    print(f"  - Total time: {total_time:.3f}s")
    print(f"  - Average frame time: {avg_frame_time*1000:.2f}ms")
    print(f"  - Std frame time: {std_frame_time*1000:.2f}ms")
    print(f"  - Min frame time: {min_frame_time*1000:.2f}ms")
    print(f"  - Max frame time: {max_frame_time*1000:.2f}ms")
    print(f"  - Achieved FPS: {fps:.2f}")
    print(f"  - Theoretical max FPS: {1.0/min_frame_time:.2f}" if min_frame_time > 0 else "")
    
    # Check for compilation issues
    if use_compile and timings:
        # Check if there's high variance (might indicate recompilation)
        cv = std_frame_time / avg_frame_time if avg_frame_time > 0 else 0
        if cv > 0.1:
            print(f"\n⚠ High coefficient of variation ({cv:.2%}) - may indicate recompilation or instability")
        
        # Check if performance degraded vs expected
        if avg_frame_time > 0.05:  # > 50ms per frame
            print(f"⚠ Slow frame times detected - compile may not be optimizing correctly")
    
    print("=" * 80)
    
    # Save output if requested
    if output_path:
        save_video(frames, output_path, fps=30)
    
    results = {
        'use_compile': use_compile,
        'compile_mode': compile_mode if use_compile else None,
        'use_inference_mode': use_inference_mode,
        'n_frames': n_frames,
        'measured_frames': measured_frames,
        'n_steps': n_steps,
        'cfg': cfg,
        'burn_in_frames': burn_in_frames,
        'total_time': total_time,
        'avg_frame_time': avg_frame_time,
        'std_frame_time': std_frame_time,
        'min_frame_time': min_frame_time,
        'max_frame_time': max_frame_time,
        'fps': fps,
        'load_time': load_time,
        'compile_time': compile_time,
        'compile_step': compile_step,
        'step_compile_time': step_compile_time if compile_step else 0,
        'timings': timings
    }
    
    return results, frames


def run_all_configurations(n_frames=500, n_steps=4, cfg=0.0, burn_in_frames=10, 
                          output_dir=None, verbose=True):
    """Run benchmarks for all configurations."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(project_root) / "experiments" / "bench" / timestamp
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Define configurations to test
    # (use_compile, compile_mode, use_inference_mode, compile_step, name)
    configurations = [
        # Eager mode (baseline)
        (False, None, True, False, "eager_inference"),
        (False, None, False, False, "eager_no_grad"),
        # Compiled model only (step function not compiled) - original behavior
        (True, "default", True, False, "compile_model_only_inference"),
        (True, "reduce-overhead", True, False, "compile_model_reduce_overhead_inference"),
        (True, "max-autotune", True, False, "compile_model_max_autotune_inference"),
        # Compiled model + step function (full optimization) - NEW OPTIMIZATION
        (True, "default", True, True, "compile_full_default_inference"),
        (True, "reduce-overhead", True, True, "compile_full_reduce_overhead_inference"),
        (True, "max-autotune", True, True, "compile_full_max_autotune_inference"),
    ]
    
    all_results = []
    
    for use_compile, compile_mode, use_inference_mode, compile_step, name in configurations:
        print("\n" + "=" * 80)
        print(f"Running configuration: {name}")
        print("=" * 80)
        
        # Create output path for this configuration
        gif_path = output_dir / f"{name}.gif"
        
        try:
            results, frames = benchmark_model(
                use_compile=use_compile,
                compile_mode=compile_mode or "default",
                use_inference_mode=use_inference_mode,
                compile_step=compile_step,
                n_frames=n_frames,
                n_steps=n_steps,
                cfg=cfg,
                burn_in_frames=burn_in_frames,
                output_path=str(gif_path),
                verbose=verbose
            )
            
            results['config_name'] = name
            all_results.append(results)
            
            # Small delay between runs
            time.sleep(1)
            
        except Exception as e:
            print(f"ERROR in configuration {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary JSON
    summary_path = output_dir / "results.json"
    # Convert timings to lists for JSON serialization (or remove them)
    summary_results = []
    for r in all_results:
        r_copy = r.copy()
        # Remove timings list for summary (keep only stats)
        r_copy.pop('timings', None)
        summary_results.append(r_copy)
    
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Configuration':<35} {'FPS':>10} {'Avg Time (ms)':>15} {'Std (ms)':>12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['config_name']:<35} {r['fps']:>10.2f} {r['avg_frame_time']*1000:>15.2f} {r['std_frame_time']*1000:>12.2f}")
    print("=" * 80)
    print(f"\nFull results saved to: {summary_path}")
    print(f"GIFs saved to: {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark torch.compile with pong model')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--no-compile', action='store_true', help='Do not use torch.compile (default)')
    parser.add_argument('--compile-mode', type=str, default='default', 
                       choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                       help='torch.compile mode')
    parser.add_argument('--no-inference-mode', action='store_true', 
                       help='Use no_grad() instead of inference_mode()')
    parser.add_argument('--frames', type=int, default=500, help='Number of frames to generate')
    parser.add_argument('--steps', type=int, default=4, help='Number of diffusion steps per frame')
    parser.add_argument('--cfg', type=float, default=0.0, help='CFG scale')
    parser.add_argument('--burn-in', type=int, default=10, help='Number of burn-in frames')
    parser.add_argument('--output', type=str, default=None, help='Output path for video (GIF or directory)')
    parser.add_argument('--compare', action='store_true', help='Run both compiled and non-compiled versions')
    parser.add_argument('--all', action='store_true', help='Run all configurations and save to experiments/bench/timestamp')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for --all mode')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.all:
        # Run all configurations
        run_all_configurations(
            n_frames=args.frames,
            n_steps=args.steps,
            cfg=args.cfg,
            burn_in_frames=args.burn_in,
            output_dir=args.output_dir,
            verbose=verbose
        )
        return
    
    if args.compare:
        # Run both versions
        print("\n" + "=" * 80)
        print("COMPARISON MODE: Running both compiled and non-compiled versions")
        print("=" * 80)
        
        # Non-compiled first
        results_no_compile, _ = benchmark_model(
            use_compile=False,
            compile_mode="default",
            use_inference_mode=not args.no_inference_mode,
            compile_step=False,
            n_frames=args.frames,
            n_steps=args.steps,
            cfg=args.cfg,
            burn_in_frames=args.burn_in,
            output_path=None,  # Don't save intermediate
            verbose=verbose
        )
        
        # Small delay between runs
        time.sleep(2)
        
        # Compiled version
        results_compile, frames = benchmark_model(
            use_compile=True,
            compile_mode=args.compile_mode,
            use_inference_mode=not args.no_inference_mode,
            compile_step=args.compile_step,
            n_frames=args.frames,
            n_steps=args.steps,
            cfg=args.cfg,
            burn_in_frames=args.burn_in,
            output_path=args.output,
            verbose=verbose
        )
        
        # Comparison summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        speedup = results_no_compile['fps'] / results_compile['fps'] if results_compile['fps'] > 0 else 0
        print(f"Without compile: {results_no_compile['fps']:.2f} FPS")
        print(f"With compile:    {results_compile['fps']:.2f} FPS")
        if speedup > 1:
            print(f"Speedup: {speedup:.2f}x faster with compile")
        elif speedup < 1:
            print(f"Slowdown: {1/speedup:.2f}x slower with compile")
        else:
            print("No significant difference")
        print("=" * 80)
    else:
        # Run single version
        use_compile = args.compile and not args.no_compile
        
        # Auto-create output directory if not specified
        output_path = args.output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"{'compile' if use_compile else 'eager'}_{args.compile_mode if use_compile else 'none'}"
            if args.no_inference_mode:
                config_name += "_no_grad"
            else:
                config_name += "_inference"
            output_dir = Path(project_root) / "experiments" / "bench" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{config_name}.gif")
            print(f"\nAuto-saving GIF to: {output_path}")
        
        benchmark_model(
            use_compile=use_compile,
            compile_mode=args.compile_mode,
            use_inference_mode=not args.no_inference_mode,
            compile_step=args.compile_step,
            n_frames=args.frames,
            n_steps=args.steps,
            cfg=args.cfg,
            burn_in_frames=args.burn_in,
            output_path=output_path,
            verbose=verbose
        )


if __name__ == '__main__':
    main()

