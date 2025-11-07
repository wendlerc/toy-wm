#!/usr/bin/env python3
"""
Gradio-based Neural Pong Game for Hugging Face Spaces
A diffusion model-powered Pong game with real-time keyboard controls
"""
import sys
import os
import time
import threading
import queue
from contextlib import contextmanager
from io import BytesIO

import torch as t
import numpy as np
from PIL import Image
import gradio as gr

# --------------------------
# Project imports
# --------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.checkpoint import load_model_from_config
from src.trainers.diffusion_forcing import sample
from src.datasets.pong1m import get_loader
from src.config import Config

# --------------------------
# Globals
# --------------------------
model = None
pred2frame = None
device = None
server_ready = False

stream_lock = threading.Lock()
stream_thread = None
stream_running = False
latest_action = 1  # 0=START, 1=NOOP, 2=UP, 3=DOWN
frame_index = 0
frame_queue = queue.Queue(maxsize=2)  # Buffer for generated frames

action_buf = None
step_once = None

# --------------------------
# Helper functions
# --------------------------
def _ensure_cuda():
    if not t.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    return t.device("cuda:0")

@contextmanager
def log_step_debug(**kwargs):
    """Debug context to log state when errors occur."""
    try:
        yield
    except Exception as ex:
        msg = f"[GEN ERROR] Step failed\n"
        for k, v in kwargs.items():
            if isinstance(v, t.Tensor):
                msg += f"  - {k}: {v.shape} | {v.dtype} | {v.device}\n"
            else:
                msg += f"  - {k}: {v}\n"
        msg += f"  - exception: {type(ex).__name__}: {ex}\n"
        import traceback as tb
        msg += f"  - trace: {tb.format_exc()}"
        raise RuntimeError(msg) from ex

# --------------------------
# Model initialization
# --------------------------
def initialize_model():
    """Initialize the model in a background thread."""
    global model, pred2frame, device, action_buf, step_once, server_ready
    
    try:
        print("🎮 Loading Neural Pong model...")
        device = _ensure_cuda()
        
        config_path = os.path.join(project_root, "configs/inference.yaml")
        cfg = Config.from_yaml(config_path)
        checkpoint_path = cfg.model.checkpoint
        
        # Handle HF Hub paths (format: hf://username/repo/file.pt)
        if checkpoint_path.startswith("hf://"):
            print(f"📥 Downloading checkpoint from Hugging Face Hub...")
            from huggingface_hub import hf_hub_download
            parts = checkpoint_path[5:].split("/", 2)  # Split into username, repo, file
            if len(parts) >= 3:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = parts[2]
                checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
                print(f"✅ Downloaded to: {checkpoint_path}")
            else:
                raise ValueError(f"Invalid HF Hub path: {checkpoint_path}")
        
        # Load model
        model = load_model_from_config(config_path, checkpoint_path=checkpoint_path, strict=False)
        model.to(device)
        model.eval()
        model.activate_caching(1, 300)
        
        # Get frame converter
        _, pred2frame_ = get_loader(duration=1, fps=30, mode='-1,1')
        pred2frame = pred2frame_
        
        # Pre-allocate buffers
        action_buf = t.empty((1, 1), dtype=t.long, device=device)
        
        # Compile step function
        @t.inference_mode()
        def _step(action_scalar: int, n_steps: int = 4, cfg: float = 0.0, clamp: bool = True):
            global model, action_buf
            
            # Prepare action
            action_buf.fill_(action_scalar)
            
            # Generate fresh noise for each step
            noise = t.randn(1, 1, 3, 24, 24, device=device, dtype=t.float32)
            
            # Run diffusion
            with t.autocast(device_type="cuda", dtype=t.bfloat16):
                z = sample(model, noise, action_buf, num_steps=n_steps, 
                          cfg=cfg, negative_actions=None)
            
            if clamp:
                z = t.clamp(z, -1.0, 1.0)
            
            return z
        
        step_once = _step
        
        # Warmup
        print("🔥 Warming up model...")
        for _ in range(3):
            z = step_once(1, n_steps=4, cfg=0.0, clamp=True)
            _ = pred2frame(z)
        
        server_ready = True
        print("✅ Neural Pong is ready!")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()

# --------------------------
# Frame generation loop
# --------------------------
class FrameGenerator:
    """Background thread that generates frames continuously."""
    
    def __init__(self, fps=4, n_steps=4, cfg=0.0, clamp=True):
        self.fps = fps
        self.n_steps = n_steps
        self.cfg = cfg
        self.clamp = clamp
        self.frame_period = 1.0 / fps
        self._stop = threading.Event()
        self._thread = None
    
    def start(self):
        """Start the frame generation thread."""
        global frame_index, model
        if self._thread is not None and self._thread.is_alive():
            return
        
        # Reset cache and frame counter
        if model is not None:
            model.cache.reset()
        frame_index = 0
        
        # Clear queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the frame generation thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def run(self):
        """Main generation loop."""
        global frame_index, latest_action
        
        next_tick = time.perf_counter()
        
        while not self._stop.is_set():
            start = time.perf_counter()
            
            # Skip if we're behind
            if start - next_tick > self.frame_period * 0.75:
                next_tick = start + self.frame_period
                continue
            
            try:
                # Get current action
                with stream_lock:
                    action = int(latest_action)
                
                # Generate frame
                with t.inference_mode():
                    with log_step_debug(action_tensor=action_buf, frame_index=frame_index):
                        z = step_once(action, n_steps=self.n_steps, 
                                    cfg=self.cfg, clamp=self.clamp)
                
                # Convert to image
                frames_btchw = pred2frame(z)
                frame_arr = frames_btchw[0, 0].permute(1, 2, 0).contiguous()
                frame_np = frame_arr.to("cpu", non_blocking=True).numpy()
                
                # Create PIL Image
                img = Image.fromarray(frame_np, mode='RGB')
                
                # Put in queue (non-blocking)
                try:
                    frame_queue.put_nowait((img, action, frame_index))
                except queue.Full:
                    pass  # Drop frame if queue is full
                
                frame_index += 1
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
                import traceback
                traceback.print_exc()
            
            # Sleep until next frame
            next_tick += self.frame_period
            now = time.perf_counter()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)

# Global generator instance
generator = None

# --------------------------
# Gradio Interface Functions
# --------------------------
def start_game():
    """Start the game stream."""
    global generator, stream_running
    
    if not server_ready:
        return None, "⏳ Model is still loading... Please wait."
    
    if generator is None:
        generator = FrameGenerator(fps=4, n_steps=4, cfg=0.0, clamp=True)
    
    generator.start()
    stream_running = True
    
    return None, "🎮 Game started! Use ↑/↓ or W/S to control the paddle."

def stop_game():
    """Stop the game stream."""
    global generator, stream_running
    
    if generator is not None:
        generator.stop()
    
    stream_running = False
    return None, "⏸️ Game stopped."

def get_frame():
    """Get the latest frame from the queue."""
    try:
        img, action, idx = frame_queue.get(timeout=0.5)
        action_labels = ['START', 'NOOP', 'UP', 'DOWN']
        action_text = f"Action: {action} ({action_labels[action] if action < len(action_labels) else 'UNKNOWN'})"
        return img, action_text
    except queue.Empty:
        # Return a black frame if no frame is available
        return Image.new('RGB', (192, 192), color='black'), "Waiting for frames..."

def update_action(key_code):
    """Update the current action based on keyboard input."""
    global latest_action
    
    # key_code mapping:
    # 38 = ArrowUp, 40 = ArrowDown
    # 87 = W, 83 = S
    
    with stream_lock:
        if key_code in [38, 87]:  # Up
            latest_action = 2
        elif key_code in [40, 83]:  # Down
            latest_action = 3
        else:  # Release or unknown
            latest_action = 1
    
    return f"Action updated: {latest_action}"

# --------------------------
# Gradio UI
# --------------------------
def create_ui():
    """Create the Gradio interface."""
    
    # Custom CSS
    css = """
    #game-canvas {
        width: 512px;
        height: 512px;
        image-rendering: pixelated;
        image-rendering: crisp-edges;
    }
    #action-display {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }
    .game-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 20px;
    }
    """
    
    # JavaScript for keyboard controls
    js_code = """
    function setupKeyboard() {
        let actionState = 1; // Start with NOOP
        
        document.addEventListener('keydown', (e) => {
            let keyCode = e.keyCode || e.which;
            if ([38, 40, 87, 83].includes(keyCode)) {
                e.preventDefault();
                // Trigger action update through Gradio
                window.dispatchEvent(new CustomEvent('pong-key', {detail: keyCode}));
            }
        });
        
        document.addEventListener('keyup', (e) => {
            let keyCode = e.keyCode || e.which;
            if ([38, 40, 87, 83].includes(keyCode)) {
                e.preventDefault();
                // Release = NOOP
                window.dispatchEvent(new CustomEvent('pong-key', {detail: 0}));
            }
        });
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupKeyboard);
    } else {
        setupKeyboard();
    }
    """
    
    with gr.Blocks(css=css, head=js_code, title="Neural Pong 🎮") as demo:
        gr.Markdown("""
        # 🎮 Neural Pong - Diffusion Model Powered Game
        
        This is a Pong game where **every frame is generated by a diffusion model** in real-time!
        The model learns to predict future game states based on your actions.
        
        ### Controls:
        - **↑ / W** - Move paddle UP
        - **↓ / S** - Move paddle DOWN
        
        ### How it works:
        - A causal diffusion transformer generates each frame conditioned on your action
        - KV-caching enables real-time inference (~4 FPS on GPU)
        - The model was trained on 1M Pong frames
        """)
        
        with gr.Row():
            with gr.Column():
                # Game display
                game_img = gr.Image(
                    label="Game Screen",
                    elem_id="game-canvas",
                    type="pil",
                    interactive=False,
                    show_label=False
                )
                
                # Action display
                action_display = gr.Textbox(
                    label="Current Action",
                    value="Press Start to begin",
                    elem_id="action-display",
                    interactive=False
                )
                
                # Controls
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Game", variant="primary")
                    stop_btn = gr.Button("⏸️ Stop Game", variant="secondary")
                
                # Status
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready! Click 'Start Game' to begin.",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("""
                ### 📊 Technical Details
                
                **Model Architecture:**
                - Diffusion Transformer (DiT)
                - 8 layers, 12 attention heads
                - 384 hidden dimensions
                - Causal KV-caching for efficiency
                
                **Training:**
                - 1M frames from Pong gameplay
                - Diffusion forcing objective
                - Learned action-conditioned dynamics
                
                **Inference:**
                - 4 diffusion steps per frame
                - bfloat16 precision
                - ~4 FPS on modern GPU
                
                ### 🚀 Try it yourself:
                1. Click "Start Game"
                2. Use arrow keys or W/S to control
                3. Watch the model generate frames in real-time!
                
                **Note:** The game may take 30-60 seconds to load the model on first run.
                """)
        
        # Event handlers
        start_btn.click(
            fn=start_game,
            inputs=[],
            outputs=[game_img, status_text]
        )
        
        stop_btn.click(
            fn=stop_game,
            inputs=[],
            outputs=[game_img, status_text]
        )
        
        # Periodic update for streaming frames
        # Use Gradio's demo.load with every parameter for auto-refresh
        demo.load(
            fn=get_frame,
            inputs=[],
            outputs=[game_img, action_display],
            every=0.25  # Update every 250ms = 4 FPS
        )
    
    return demo

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Start model loading in background
    init_thread = threading.Thread(target=initialize_model, daemon=True)
    init_thread.start()
    
    # Create and launch Gradio app
    demo = create_ui()
    demo.queue()  # Enable queuing for better performance
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

