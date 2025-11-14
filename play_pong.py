#!/usr/bin/env python3
"""
Pong backend (GPU, eager). Now broadcasts readiness via Socket.IO so the
frontend can auto-hide a loading overlay once the model is ready.
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
import shutil
import tempfile
from contextlib import contextmanager
from io import BytesIO

import torch as t
import torch._dynamo as _dynamo
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# --------------------------
# Project imports
# --------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.checkpoint import load_model_from_config
from src.trainers.diffusion_forcing import sample
from src.datasets.pong1m import get_loader, fixed2frame
from src.config import Config

# --------------------------
# App setup
# --------------------------
app = Flask(__name__, static_folder='static')
CORS(app)
# Configure SocketIO - use eventlet for proper WebSocket support
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet', 
    logger=False, 
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8  # Allow larger messages
)

# --------------------------
# Globals
# --------------------------
model = None
pred2frame = None
device = None
cache = None

server_ready = False    # <--- readiness flag

# Single-user limitation
active_user_sid = None  # Session ID of the active user
user_lock = threading.Lock()  # Protects active_user_sid

stream_lock = threading.Lock()
stream_thread = None
stream_running = False
latest_action = 1  # 0=init, 1=nothing, 2=up, 3=down
target_fps = 30
frame_index = 0

noise_buf = None       # (1,1,3,24,24) on GPU
action_buf = None      # (1,1) long on GPU
cpu_png_buffer = None  # BytesIO; reused

step_once = None

# --------------------------
# Perf (new API)
# --------------------------
t.backends.cudnn.benchmark = True
t.backends.cudnn.conv.fp32_precision = "tf32"
t.backends.cuda.matmul.fp32_precision = "high"

# --------------------------
# Debug helpers
# --------------------------
def _shape(x):
    try:
        return f"{tuple(x.shape)} | {x.dtype} | {x.device}"
    except Exception:
        return "<?>"

def _shape_attr(obj, name):
    try:
        ten = getattr(obj, name, None)
        return None if ten is None else _shape(ten)
    except Exception:
        return None

def _fail(msg, extra=None):
    lines = [f"[GEN ERROR] {msg}"]
    if extra:
        for k, v in extra.items():
            lines.append(f"  - {k}: {v}")
    raise RuntimeError("\n".join(lines))

@contextmanager
def log_step_debug(action_tensor=None, noise_tensor=None):
    try:
        yield
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        _fail("Step failed",
              extra={
                  "action": _shape(action_tensor),
                  "noise": _shape(noise_tensor),
                  "model.device": str(device),
                  "cache.keys": _shape_attr(getattr(model, "cache", None), "keys"),
                  "cache.values": _shape_attr(getattr(model, "cache", None), "values"),
                  "frame_index": str(frame_index),
                  "exception": f"{type(e).__name__}: {e}",
                  "trace": tb.strip()
              })

# --------------------------
# Utilities
# --------------------------
def _ensure_cuda():
    if not t.cuda.is_available():
        raise RuntimeError("CUDA GPU required; torch.cuda.is_available() is False.")
    return t.device("cuda:0")

def _png_base64_from_uint8(frame_uint8) -> str:
    global cpu_png_buffer
    if cpu_png_buffer is None:
        cpu_png_buffer = BytesIO()
    else:
        cpu_png_buffer.seek(0)
        cpu_png_buffer.truncate(0)
    Image.fromarray(frame_uint8).save(cpu_png_buffer, format="PNG")
    return base64.b64encode(cpu_png_buffer.getvalue()).decode()



def _reset_cache_fresh():
    cache.reset()

def _broadcast_ready():
    """Tell all clients whether the server is ready."""
    socketio.emit('server_status', {'ready': server_ready})

# --------------------------
# Model init (pure eager) & warmup
# --------------------------
def initialize_model():
    global model, pred2frame, device, cache
    global noise_buf, action_buf, step_once, server_ready

    t_start = time.time()
    print("Loading model and preparing GPU runtime...")
    device = _ensure_cuda()

    config_path = os.path.join(project_root, "configs/inference.yaml")
    
    # Optimize checkpoint loading: copy to local storage if on network mount
    t0 = time.time()
    cfg = Config.from_yaml(config_path)
    checkpoint_path = cfg.model.checkpoint
    
    model = load_model_from_config(config_path, checkpoint_path=checkpoint_path, strict=False)
    model.to(device)  # Move model to GPU before activating cache
    model.eval()
    
    cache = model.create_cache(1)  # Cache will now be created on the same device as model
    
    # Configure dynamo to prevent recompilation from cache state changes
    # allow_unspec_int_on_nn_module prevents recompilation when cache pointer attributes
    # (local_loc, _write_ptr) change between frames
    t._dynamo.config.allow_unspec_int_on_nn_module = True
    t._dynamo.config.cache_size_limit = 128  # Increased to handle cache state changes
    
    model = t.compile(model)


    _, pred2frame_ = get_loader(duration=1, fps=30)
    globals()["pred2frame"] = pred2frame_

    H = W = 24
    noise_buf = t.empty((1, 1, 3, H, W), device=device)
    action_buf = t.empty((1, 1), dtype=t.long, device=device)

    @_dynamo.disable
    def _step(model_, action_scalar_long: int, n_steps: int, cfg: float, clamp: bool, cache=cache):
        # Match the notebook logic exactly: create fresh noise each time
        noise = t.randn(1, 1, 3, 24, 24, device=device)
        action_buf.fill_(int(action_scalar_long))

        assert action_buf.shape == (1, 1) and action_buf.dtype == t.long and action_buf.device == device, \
            f"action_buf wrong: { _shape(action_buf) }"
        assert noise.shape == (1, 1, 3, 24, 24) and noise.device == device, \
            f"noise wrong: { _shape(noise) }"

        # Debug: Check cache state before sampling
        if cache is not None:
            cache_loc = cache.local_location
            if cache_loc == 0:
                # Cache is empty, this should be fine for the first frame
                pass
            elif cache_loc > 0:
                # Check if cache has valid data
                k_test, v_test = cache.get()
                if k_test.shape[2] == 0:
                    print(f"Warning: Cache returned empty tensors at frame {frame_index}, resetting...")
                    _reset_cache_fresh()

        # Sample with the fresh noise (matching notebook: sample(model, noise, actions[:, aidx:aidx+1], ...))
        z = sample(model_, noise, action_buf, num_steps=n_steps, cfg=cfg, negative_actions=None, cache=cache)
        
        if clamp:
            z = t.clamp(z, -1, 1)
        return z

    globals()["step_once"] = _step

    # Warmup - need enough frames to trigger cache wrap-around to prevent recompilation at frame 30
    # Cache wraps after n_window frames (30), so we warm up with 35+ frames to ensure both
    # code paths in cache.get() (contiguous slice and wrap-around concatenation) are compiled
    _reset_cache_fresh()
    print("Warming up model (including cache wrap-around path)...")
    with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
        for i in range(35):  # Warm up enough to trigger cache wrap-around
            with log_step_debug(action_tensor=action_buf, noise_tensor=noise_buf):
                _ = step_once(model, action_scalar_long=1, n_steps=4, cfg=0.0, clamp=True)
            if i == 29:
                print(f"  Frame 30: Cache wrap-around should occur here, ensuring both code paths are compiled")
    print("Warmup complete")

    server_ready = True
    print(f"Model ready on {device}")
    _broadcast_ready()
    return model, pred2frame

# --------------------------
# Fixed-FPS streaming worker
# --------------------------
class FrameScheduler(threading.Thread):
    def __init__(self, fps=30, n_steps=8, cfg=0.0, clamp=True):
        super().__init__(daemon=True)
        self.frame_period = 1.0 / max(1, int(fps))
        self.n_steps = int(n_steps)
        self.cfg = float(cfg)
        self.clamp = bool(clamp)
        self._stop = threading.Event()
        # FPS tracking
        self.frame_times = []
        self.last_frame_time = None

    def stop(self):
        self._stop.set()

    def run(self):
        global frame_index, latest_action
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            start = time.perf_counter()
            if start - next_tick > self.frame_period * 0.75:
                next_tick = start + self.frame_period
                continue
            try:
                with stream_lock:
                    action = int(latest_action)
                with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
                    with log_step_debug(action_tensor=action_buf, noise_tensor=noise_buf):
                        z = step_once(model, action_scalar_long=action,
                                      n_steps=self.n_steps, cfg=self.cfg, clamp=self.clamp)
                frames_btchw = pred2frame(z)
                # Debug: check what pred2frame returns
                if frame_index < 3:
                    print(f"Frame {frame_index}: z range [{z.min().item():.3f}, {z.max().item():.3f}], "
                          f"frames_btchw dtype={frames_btchw.dtype}, range [{frames_btchw.min().item()}, {frames_btchw.max().item()}]")
                
                frame_arr = frames_btchw[0, 0].permute(1, 2, 0).contiguous()
                if isinstance(frame_arr, t.Tensor):
                    frame_np = frame_arr.to("cpu", non_blocking=True).numpy()
                else:
                    frame_np = frame_arr.astype(np.uint8, copy=False)
                img_b64 = _png_base64_from_uint8(frame_np)
                
                # Calculate achieved FPS
                current_time = time.perf_counter()
                if self.last_frame_time is not None:
                    frame_delta = current_time - self.last_frame_time
                    self.frame_times.append(frame_delta)
                    # Keep only last 30 frames for moving average
                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    achieved_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                else:
                    achieved_fps = 0
                self.last_frame_time = current_time
                
                socketio.emit('frame', {'frame': img_b64,
                                        'frame_index': frame_index,
                                        'action': action,
                                        'fps': achieved_fps})
                frame_index += 1
            except Exception as e:
                print("Generation error:", repr(e))
                socketio.emit('error', {'message': str(e)})
            next_tick += self.frame_period
            now = time.perf_counter()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)

# --------------------------
# Routes
# --------------------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.errorhandler(500)
def handle_500(e):
    """Handle WSGI errors gracefully"""
    import traceback
    print(f"Flask error handler caught: {e}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'ready': server_ready,
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'stream_running': stream_running,
        'target_fps': target_fps
    })

@app.route('/api/generate', methods=['POST'])
def generate_frames():
    try:
        if not server_ready:
            return jsonify({'success': False, 'error': 'Server not ready'}), 503

        data = request.json or {}
        actions_list = data.get('actions', [1])
        n_steps = int(data.get('n_steps', 8))
        cfg = float(data.get('cfg', 0))
        clamp = bool(data.get('clamp', True))

        if len(actions_list) == 0 or actions_list[0] != 0:
            actions_list = [0] + actions_list

        _reset_cache_fresh()

        frames_png = []
        with t.inference_mode(), t.autocast(device_type="cuda", dtype=t.bfloat16):
            for a in actions_list:
                with log_step_debug(action_tensor=action_buf, noise_tensor=noise_buf):
                    z = step_once(model, action_scalar_long=int(a), n_steps=n_steps, cfg=cfg, clamp=clamp)
                f_btchw = pred2frame(z)
                f_arr = f_btchw[0, 0].permute(1, 2, 0).contiguous()
                if isinstance(f_arr, t.Tensor):
                    if f_arr.dtype != t.uint8:
                        f_arr = f_arr.to(t.uint8)
                    f_np = f_arr.to("cpu", non_blocking=True).numpy()
                else:
                    f_np = f_arr.astype(np.uint8, copy=False)
                frames_png.append(_png_base64_from_uint8(f_np))

        return jsonify({'success': True, 'frames': frames_png, 'num_frames': len(frames_png)})

    except Exception as e:
        print("Batch generation error:", repr(e))
        return jsonify({'success': False, 'error': str(e)}), 500

# --------------------------
# Socket events & helpers
# --------------------------
def start_stream(n_steps=8, cfg=0.0, fps=30, clamp=True):
    global stream_thread, stream_running, frame_index, target_fps, latest_action
    if not server_ready:
        _broadcast_ready()
        raise RuntimeError("Server not ready")
    with stream_lock:
        stop_stream()
        target_fps = min(60, int(fps))
        frame_index = 0
        _reset_cache_fresh()
        latest_action = 0  # first action = 0 (init)
        stream_thread = FrameScheduler(fps=target_fps, n_steps=min(10, n_steps), cfg=cfg, clamp=clamp)
        stream_running = True
        stream_thread.start()

def stop_stream():
    global stream_thread, stream_running
    if stream_thread is not None:
        stream_thread.stop()
        stream_thread.join(timeout=1.0)
        stream_thread = None
    stream_running = False

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO error: {e}")
    import traceback
    traceback.print_exc()

@socketio.on('connect')
def handle_connect():
    try:
        sid = request.sid
        print(f'Client connected: {sid}')
        
        with user_lock:
            is_busy = active_user_sid is not None and active_user_sid != sid
        
        # Immediately tell the new client current readiness and availability
        emit('server_status', {
            'ready': server_ready,
            'busy': is_busy,
            'is_active_user': not is_busy
        })
        emit('connected', {
            'status': 'connected',
            'model_loaded': model is not None,
            'ready': server_ready,
            'busy': is_busy
        })
    except Exception as e:
        print(f"Error in handle_connect: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect(*args):
    global active_user_sid
    sid = request.sid
    print(f'Client disconnected: {sid}')
    
    # Release the active user slot if this was the active user
    with user_lock:
        if active_user_sid == sid:
            print(f'Active user {sid} disconnected, freeing slot')
            active_user_sid = None
            # Notify all other clients that server is now available
            socketio.emit('server_status', {
                'ready': server_ready,
                'busy': False,
                'is_active_user': False
            })
    
    stop_stream()

@socketio.on('start_stream')
def handle_start_stream(data):
    global active_user_sid
    try:
        sid = request.sid
        
        if not server_ready:
            # Tell client to keep showing spinner
            emit('server_status', {'ready': server_ready})
            return
        
        # Check if server is busy with another user
        with user_lock:
            if active_user_sid is not None and active_user_sid != sid:
                emit('error', {'message': 'Server is currently being used by another user. Please wait.'})
                emit('server_status', {
                    'ready': server_ready,
                    'busy': True,
                    'is_active_user': False
                })
                return
            # Claim the active user slot
            active_user_sid = sid
            print(f'User {sid} claimed active slot')
        
        # Notify all clients about the new busy state
        socketio.emit('server_status', {
            'ready': server_ready,
            'busy': True,
            'is_active_user': False
        }, include_self=False)
        
        n_steps = min(10, int(data.get('n_steps', 8)))
        cfg = float(data.get('cfg', 0))
        fps = min(60, int(data.get('fps', 30)))
        clamp = bool(data.get('clamp', True))
        print(f"Starting stream @ {fps} FPS (n_steps={n_steps}, cfg={cfg}, clamp={clamp})")
        try:
            start_stream(n_steps=n_steps, cfg=cfg, fps=fps, clamp=clamp)
            emit('stream_started', {'status': 'ok'})
        except Exception as e:
            print(f"Error starting stream: {e}")
            import traceback
            traceback.print_exc()
            # Release the slot on error
            with user_lock:
                if active_user_sid == sid:
                    active_user_sid = None
            emit('error', {'message': str(e)})
    except Exception as e:
        print(f"Error in handle_start_stream: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Failed to start stream: {str(e)}'})

@socketio.on('action')
def handle_action(data):
    global latest_action
    sid = request.sid
    
    # Only accept actions from the active user
    with user_lock:
        if active_user_sid != sid:
            return  # Silently ignore actions from non-active users
    
    action = int(data.get('action', 1))
    with stream_lock:
        latest_action = action
    emit('action_ack', {'received': action, 'will_apply_to_frame_index': frame_index})

@socketio.on('stop_stream')
def handle_stop_stream():
    global active_user_sid
    sid = request.sid
    
    # Only the active user can stop the stream
    with user_lock:
        if active_user_sid != sid:
            return  # Silently ignore stop requests from non-active users
        # Release the active user slot
        print(f'User {sid} stopped stream and released slot')
        active_user_sid = None
    
    # Notify all clients that server is now available
    socketio.emit('server_status', {
        'ready': server_ready,
        'busy': False,
        'is_active_user': False
    })
    
    print('Stopping stream')
    stop_stream()

# --------------------------
# Entrypoint
# --------------------------
if __name__ == '__main__':

    initialize_model()
    
    print("Starting Flask server on http://localhost:5000")
    print("Model will load in background...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True, use_reloader=False)