#!/usr/bin/env python3
"""
Interactive Doom world-model inference server (DiT only, minimal).

Uses latent diffusion with DC-AE VAE decoding. Keyboard + mouse input
via SocketIO with simultaneous keys (bitmask protocol).

Usage:
    uv run python play_doom.py
    uv run python play_doom.py --checkpoint experiments/doom-run/model.pt
    uv run python play_doom.py --config configs/doom_diffusion_forcing.yaml --port 4444
"""

# Eventlet must be imported first and monkey-patched before other imports
import eventlet
eventlet.monkey_patch()

import sys
import os
import time
import threading
import base64
import traceback
from io import BytesIO

import torch as t
import torch._dynamo as _dynamo
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import argparse

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.checkpoint import load_model_from_config
from src.inference.sampling import sample
from src.config import Config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(
    app, cors_allowed_origins="*", async_mode='eventlet',
    logger=False, engineio_logger=False,
    ping_timeout=60, ping_interval=25, max_http_buffer_size=1e8,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
model = None
device = None
cache = None
vae = None
vae_decode_compiled = None

server_ready = False
active_user_sid = None
user_lock = threading.Lock()

stream_lock = threading.Lock()
stream_thread = None
stream_running = False
target_fps = 30
frame_index = 0

# Action state
latest_keys = 0
turn_accum = 0.0
is_init_frame = True
uncond_mode = False

noise_buf = None
action_buf = None
step_once = None
seed_cache_fn = None

cpu_jpeg_buffer = None

# Start frames sampled from dataset
start_frames = []

# ---------------------------------------------------------------------------
# CUDA settings
# ---------------------------------------------------------------------------
t.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Doom action mapping
# ---------------------------------------------------------------------------
# Keyboard bitmask bits:
#   0x0001: Forward (W)    0x0002: Backward (S)
#   0x0004: Strafe Left (A) 0x0008: Strafe Right (D)
#   0x0010: Attack (Click/Space) 0x0020: Sprint (Shift)
#   0x0040-0x1000: Weapons 1-7
#
# Action vector: [uncond, MF, MB, MR, ML, W1-W7, ATK, SPD, TURN]

def _uncond_action():
    return [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def _build_action(keys: int, turn: float) -> list[float]:
    action = [0.0] * 15
    if keys & 0x0001: action[1] = 1.0   # Forward
    if keys & 0x0002: action[2] = 1.0   # Backward
    if keys & 0x0004: action[4] = 1.0   # Strafe Left
    if keys & 0x0008: action[3] = 1.0   # Strafe Right
    if keys & 0x0010: action[12] = 1.0  # Attack
    if keys & 0x0020: action[13] = 1.0  # Sprint
    if keys & 0x0040: action[5] = 1.0   # W1
    if keys & 0x0080: action[6] = 1.0   # W2
    if keys & 0x0100: action[7] = 1.0   # W3
    if keys & 0x0200: action[8] = 1.0   # W4
    if keys & 0x0400: action[9] = 1.0   # W5
    if keys & 0x0800: action[10] = 1.0  # W6
    if keys & 0x1000: action[11] = 1.0  # W7
    action[14] = max(-12.5, min(12.5, turn))
    return action

# ---------------------------------------------------------------------------
# VAE decoder
# ---------------------------------------------------------------------------
def _load_vae(dev):
    from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
    import glob
    local = glob.glob("/tmp/dc-ae-cache/snapshots/*/")
    model_path = local[0] if local else "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
    print(f"  Loading VAE decoder from {model_path}")
    return AutoencoderDC.from_pretrained(model_path, torch_dtype=t.float16).to(dev).eval()

def _fast_vae_decode(z_latent):
    """Decode latent (1,1,32,16,20) -> RGB uint8 (1,3,480,640)."""
    flat = z_latent[0, :, :, 1:, :].to(t.float16)  # strip height padding
    rgb = vae_decode_compiled(flat).sample
    return ((rgb.clamp(-1, 1) + 1) / 2 * 255).byte()

# ---------------------------------------------------------------------------
# Start frame sampling from dataset
# ---------------------------------------------------------------------------
def _sample_start_frames_from_dataset(shard_dir, n_window, n_frames=12):
    """Sample random clips from dataset shards for the start frame picker."""
    import random
    import tarfile
    import glob as globmod
    shards = sorted(globmod.glob(os.path.join(shard_dir, "latent-*.tar")))
    if not shards:
        print(f"  No dataset shards in {shard_dir}")
        return []

    picked = random.sample(shards, min(len(shards), n_frames))
    clips = []

    for shard_path in picked:
        try:
            with tarfile.open(shard_path) as tf:
                members = tf.getnames()
                eps = set()
                for name in members:
                    if name.endswith('.latents_p1.npy'):
                        eps.add(name.rsplit('.latents_p1.npy', 1)[0])
                if not eps:
                    continue
                ep = random.choice(list(eps))

                lat_f = tf.extractfile(f"{ep}.latents_p1.npy")
                lat_raw = np.load(BytesIO(lat_f.read()))
                act_f = tf.extractfile(f"{ep}.actions_p1.npy")
                act_raw = np.load(BytesIO(act_f.read()))

                n_total = min(lat_raw.shape[0], act_raw.shape[0])
                if n_total < n_window + 1:
                    continue

                start_idx = random.randint(0, n_total - n_window - 1)
                lat_clip = lat_raw[start_idx:start_idx + n_window]
                act_clip = act_raw[start_idx:start_idx + n_window]

                # Pad height 15 -> 16 (zero first row)
                lat_padded = np.zeros((n_window, 32, 16, 20), dtype=np.float32)
                lat_padded[:, :, 1:, :] = lat_clip

                # Convert 14-dim dataset actions to 15-dim (prepend uncond=0)
                act_15 = np.zeros((n_window, 15), dtype=np.float32)
                act_15[:, 1:] = act_clip

                clips.append({
                    'latents': t.tensor(lat_padded),
                    'actions': t.tensor(act_15),
                })
        except Exception as e:
            print(f"  Warning: failed to load shard {shard_path}: {e}")
            continue

    print(f"  Sampled {len(clips)} start frames from dataset")
    return clips


def _load_start_frames(n_window, shard_dir):
    """Load start frame clips and decode thumbnails for the UI picker."""
    global start_frames
    clips = _sample_start_frames_from_dataset(shard_dir, n_window)
    if not clips:
        print("  Warning: no start frames available")
        return

    # Decode last frame of each clip to a thumbnail
    for clip in clips:
        z0 = clip['latents'][-1:].unsqueeze(0).to(device=device, dtype=t.float16)
        with t.inference_mode():
            rgb_gpu = _fast_vae_decode(z0)
        rgb_np = rgb_gpu[0].permute(1, 2, 0).cpu().numpy()
        clip['thumbnail'] = _jpeg_base64(rgb_np, quality=70)

    start_frames = clips
    print(f"  Loaded {len(start_frames)} start frames with thumbnails")


def _jpeg_base64(frame_uint8_np, quality=85) -> str:
    global cpu_jpeg_buffer
    if cpu_jpeg_buffer is None:
        cpu_jpeg_buffer = BytesIO()
    else:
        cpu_jpeg_buffer.seek(0)
        cpu_jpeg_buffer.truncate(0)
    Image.fromarray(frame_uint8_np).save(cpu_jpeg_buffer, format="JPEG", quality=quality)
    return base64.b64encode(cpu_jpeg_buffer.getvalue()).decode()

# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------
def _ensure_cuda():
    if not t.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    return t.device("cuda:0")

def _reset_cache():
    cache.reset()

def _broadcast_ready():
    socketio.emit('server_status', {'ready': server_ready})

def initialize_model(config_path, checkpoint_override=None):
    global model, device, cache, vae, vae_decode_compiled
    global noise_buf, action_buf, step_once, seed_cache_fn, server_ready

    t_start = time.time()
    print("=" * 60)
    print("Initializing Doom world model + VAE decoder")
    print("=" * 60)
    device = _ensure_cuda()

    config_path = os.path.abspath(config_path)
    cfg = Config.from_yaml(config_path)
    checkpoint_path = checkpoint_override or cfg.model.checkpoint
    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")
    else:
        print("  WARNING: No checkpoint — model has random weights.")

    cmodel = cfg.model
    C, H, W = cmodel.in_channels, cmodel.height, cmodel.width
    print(f"  Latent shape: ({C}, {H}, {W})")

    # Load diffusion model
    model = load_model_from_config(config_path, checkpoint_path=checkpoint_path, strict=False)
    model.to(device).eval()
    cache = model.create_cache(1)  # batch=1, no CFG for speed

    t._dynamo.config.allow_unspec_int_on_nn_module = True
    t._dynamo.config.cache_size_limit = 128
    if not cmodel.nocompile:
        model = t.compile(model)
        print(f"  Diffusion model compiled ({time.time()-t_start:.1f}s)")

    # Load VAE decoder
    t_vae = time.time()
    vae = _load_vae(device)
    vae_decode_compiled = t.compile(vae.decode, mode="reduce-overhead")
    print(f"  VAE decoder loaded + compiled ({time.time()-t_vae:.1f}s)")

    # Buffers
    noise_buf = t.empty((1, 1, C, H, W), device=device)
    action_buf = t.zeros((1, 1, 15), dtype=model.dtype, device=device)

    # Step function (no CFG — fast path)
    @_dynamo.disable
    def _step(model_, action_vec: list[float], n_steps: int, clamp: bool, cache=cache):
        noise = t.randn(1, 1, C, H, W, device=device, dtype=model_.dtype)
        action_buf.copy_(t.tensor(action_vec, dtype=model_.dtype, device=device).view(1, 1, 15))
        # Fast path: batch=1, no CFG
        ts = 1 - t.linspace(0, 1, n_steps + 1, device=device)
        ts = 3 * ts / (2 * ts + 1)
        z = noise.clone()
        rlimit = len(ts)  # extra_step=True
        for i in range(rlimit):
            t_cond = ts[i].reshape(1, 1)
            cached_k, cached_v = cache.get() if cache is not None else (None, None)
            v_pred, k_new, v_new = model_(z, action_buf, t_cond,
                                          cached_k=cached_k, cached_v=cached_v)
            if i < rlimit - 1:
                z = z + (ts[i] - ts[i + 1]) * v_pred
        if cache is not None:
            cache.extend(k_new, v_new)
        if clamp:
            z = t.clamp(z, -1, 1)
        return z

    step_once = _step

    # Seed function: single forward pass at t=0 with a known latent to populate KV cache
    @_dynamo.disable
    def _seed_step(model_, z_latent, action_vec_tensor, cache=cache):
        action_buf.copy_(action_vec_tensor)
        t_cond = t.zeros(1, 1, device=device)
        cached_k, cached_v = cache.get() if cache is not None else (None, None)
        _, k_new, v_new = model_(z_latent, action_buf, t_cond,
                                  cached_k=cached_k, cached_v=cached_v)
        if cache is not None:
            cache.extend(k_new, v_new)

    seed_cache_fn = _seed_step

    # Warmup
    _reset_cache()
    warmup_frames = cmodel.n_window + 5
    print(f"  Warming up ({warmup_frames} frames)...")
    with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
        for i in range(warmup_frames):
            _ = step_once(model, action_vec=_uncond_action(), n_steps=4, clamp=False)

    # Warmup seed path
    print("  Warming up seed path...")
    _reset_cache()
    dummy_z = t.randn(1, 1, C, H, W, device=device, dtype=model.dtype)
    dummy_a = t.zeros(1, 1, 15, device=device, dtype=model.dtype)
    with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
        for i in range(min(warmup_frames, cmodel.n_window)):
            seed_cache_fn(model, dummy_z, dummy_a)

    # VAE warmup
    print("  Warming up VAE decoder...")
    dummy_latent = t.randn(1, 1, C, H, W, device=device, dtype=t.bfloat16)
    with t.inference_mode():
        for _ in range(3):
            _ = _fast_vae_decode(dummy_latent)
    t.cuda.synchronize()

    # Load start frames from dataset
    print("  Loading start frames from dataset...")
    shard_dir = getattr(cfg.dataset, "shard_dir", None) or "./datasets/doom_latents"
    _load_start_frames(cmodel.n_window, shard_dir=shard_dir)
    t.cuda.synchronize()

    server_ready = True
    print(f"\nServer ready on {device} ({time.time()-t_start:.1f}s total)")
    _broadcast_ready()
    return model

# ---------------------------------------------------------------------------
# Frame streaming
# ---------------------------------------------------------------------------
class FrameScheduler(threading.Thread):
    def __init__(self, fps=30, n_steps=6, clamp=False):
        super().__init__(daemon=True)
        self.frame_period = 1.0 / max(1, int(fps))
        self.n_steps = int(n_steps)
        self.clamp = bool(clamp)
        self._stop = threading.Event()
        self.frame_times = []
        self.last_frame_time = None

    def stop(self):
        self._stop.set()

    def run(self):
        global frame_index, latest_keys, turn_accum, is_init_frame
        next_tick = time.perf_counter()

        while not self._stop.is_set():
            start = time.perf_counter()
            if start - next_tick > self.frame_period * 0.75:
                next_tick = start + self.frame_period
                continue
            try:
                with stream_lock:
                    if is_init_frame:
                        action_vec = _uncond_action()
                        action_desc = "init"
                        is_init_frame = False
                    elif uncond_mode:
                        action_vec = _uncond_action()
                        action_desc = "uncond"
                    else:
                        keys = latest_keys
                        turn = turn_accum
                        turn_accum = 0.0
                        action_vec = _build_action(keys, turn)
                        action_desc = f"keys=0x{keys:04x},turn={turn:.1f}"

                with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
                    z = step_once(model, action_vec=action_vec,
                                  n_steps=self.n_steps, clamp=self.clamp)
                    rgb_gpu = _fast_vae_decode(z)

                rgb_np = rgb_gpu[0].permute(1, 2, 0).cpu().numpy()
                img_b64 = _jpeg_base64(rgb_np)

                now = time.perf_counter()
                if self.last_frame_time is not None:
                    dt = now - self.last_frame_time
                    self.frame_times.append(dt)
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)
                    fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                else:
                    fps = 0
                self.last_frame_time = now

                socketio.emit('frame', {
                    'frame': img_b64,
                    'frame_index': frame_index,
                    'action': action_desc,
                    'fps': fps,
                })
                frame_index += 1
            except Exception as e:
                print("Generation error:", repr(e))
                traceback.print_exc()
                socketio.emit('error', {'message': str(e)})

            next_tick += self.frame_period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory('static', 'doom.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 'ready': server_ready,
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'stream_running': stream_running, 'target_fps': target_fps,
    })

# ---------------------------------------------------------------------------
# Socket events
# ---------------------------------------------------------------------------
def start_stream(n_steps=6, fps=30, clamp=False, start_frame_idx=-1):
    global stream_thread, stream_running, frame_index, target_fps
    global latest_keys, turn_accum, is_init_frame, uncond_mode
    if not server_ready:
        _broadcast_ready()
        raise RuntimeError("Server not ready")
    with stream_lock:
        stop_stream()
        target_fps = max(1, int(fps))
        frame_index = 0
        _reset_cache()
        latest_keys = 0
        turn_accum = 0.0
        uncond_mode = False

        if start_frame_idx >= 0 and start_frame_idx < len(start_frames):
            # Seed KV cache from a real gameplay clip
            clip = start_frames[start_frame_idx]
            latents = clip['latents'].to(device=device, dtype=model.dtype)
            actions = clip['actions'].to(device=device, dtype=model.dtype)
            with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
                for i in range(latents.shape[0]):
                    z = latents[i:i+1].unsqueeze(0)
                    ab = actions[i:i+1].unsqueeze(0)
                    seed_cache_fn(model, z, ab)
            is_init_frame = False
            print(f"  Seeded cache from start frame {start_frame_idx}")
        else:
            is_init_frame = True

        stream_thread = FrameScheduler(fps=target_fps, n_steps=n_steps, clamp=clamp)
        stream_running = True
        stream_thread.start()

def stop_stream():
    global stream_thread, stream_running
    if stream_thread is not None:
        stream_thread.stop()
        try:
            stream_thread.join(timeout=5.0)
        except Exception:
            pass
        stream_thread = None
    stream_running = False

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO error: {e}")
    traceback.print_exc()

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f'Client connected: {sid}')
    with user_lock:
        is_busy = active_user_sid is not None and active_user_sid != sid
    emit('server_status', {'ready': server_ready, 'busy': is_busy, 'is_active_user': not is_busy})
    emit('connected', {'status': 'connected', 'model_loaded': model is not None,
                       'ready': server_ready, 'busy': is_busy})
    # Send start frame thumbnails
    if start_frames:
        thumbs = [{'idx': i, 'thumbnail': sf['thumbnail']} for i, sf in enumerate(start_frames)]
        emit('start_frames', thumbs)

@socketio.on('disconnect')
def handle_disconnect(*args):
    global active_user_sid
    sid = request.sid
    print(f'Client disconnected: {sid}')
    with user_lock:
        if active_user_sid == sid:
            active_user_sid = None
            socketio.emit('server_status', {'ready': server_ready, 'busy': False, 'is_active_user': False})
    stop_stream()

@socketio.on('start_stream')
def handle_start_stream(data):
    global active_user_sid
    try:
        sid = request.sid
        if not server_ready:
            emit('server_status', {'ready': server_ready})
            return
        with user_lock:
            if active_user_sid is not None and active_user_sid != sid:
                emit('error', {'message': 'Server busy with another user.'})
                return
            active_user_sid = sid
        n_steps = int(data.get('n_steps', 6))
        fps = max(1, int(data.get('fps', 30)))
        clamp = bool(data.get('clamp', False))
        start_frame_idx = int(data.get('start_frame', -1))
        sf_desc = f", start_frame={start_frame_idx}" if start_frame_idx >= 0 else ""
        print(f"Starting stream @ {fps} FPS (n_steps={n_steps}, clamp={clamp}{sf_desc})")
        start_stream(n_steps=n_steps, fps=fps, clamp=clamp, start_frame_idx=start_frame_idx)
        emit('stream_started', {'status': 'ok'})
    except Exception as e:
        print(f"Error starting stream: {e}")
        traceback.print_exc()
        emit('error', {'message': str(e)})

@socketio.on('action')
def handle_action(data):
    global latest_keys, turn_accum, uncond_mode
    sid = request.sid
    with user_lock:
        if active_user_sid != sid:
            return
    with stream_lock:
        latest_keys = int(data.get('keys', 0))
        turn_accum += float(data.get('turn', 0.0))
        uncond_mode = bool(data.get('uncond', False))

@socketio.on('stop_stream')
def handle_stop_stream():
    global active_user_sid
    sid = request.sid
    with user_lock:
        if active_user_sid != sid:
            return
        active_user_sid = None
    socketio.emit('server_status', {'ready': server_ready, 'busy': False, 'is_active_user': False})
    stop_stream()

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Doom inference server")
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, "configs/doom_diffusion_forcing.yaml"))
    parser.add_argument('--port', type=int, default=4444)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Override checkpoint path (directory or .pt file)")
    args = parser.parse_args()

    initialize_model(args.config, checkpoint_override=args.checkpoint)
    print(f"Starting server on http://localhost:{args.port}")
    socketio.run(app, host='0.0.0.0', port=args.port, debug=False,
                 allow_unsafe_werkzeug=True, use_reloader=False)
