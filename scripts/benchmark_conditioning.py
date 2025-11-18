import torch as t
from torch import nn

class CondBench(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mod = nn.Linear(d, 3*d)

    def naive(self, x, cond):
        # x: b, dur*c*h*w, d
        # cond: b, dur*c*h*w, d
        scale, shift, gate = self.mod(cond).chunk(3, dim=-1)
        x = (x * scale + shift) * gate
        return x 

    def reshape_based(self, x, cond, toks_per_frame):
        # x: b, dur*c*h*w, d
        # cond: b, dur, d
        b, l, d = x.shape
        scale, shift, gate = self.mod(cond).chunk(3, dim=-1)
        x = (x.reshape(b, -1, toks_per_frame, d) * scale[:, :, None, :] + shift[:, :, None, :]) * gate[:, :, None, :]
        return x.reshape(b, l, d)

    def scatter_based(self, x, cond, toks_per_frame):
        # x: b, dur*c*h*w, d
        # cond: b, dur, d
        # will broadcast cond over c*h*w tokens for each dur
        # toks_per_frame: number of tokens per "dur"
        b, l, d = x.shape
        dur = cond.shape[1]
        # Step 1: get scale, shift, gate of shape [b, dur, d]
        scale, shift, gate = self.mod(cond).chunk(3, dim=-1)
        
        # Step 2: create the frame indices - which frame each token belongs to
        # frame_idx: (l,) - each token's frame index
        frame_idx = (t.arange(l, device=x.device) // toks_per_frame).clamp(max=dur-1)
        
        # Step 3: Use advanced indexing to gather scale/shift/gate for each token
        # scale is (b, dur, d), we want to index along dim=1 with frame_idx
        # Using: scale[b, frame_idx[i], :] for each i
        # Result: (b, l, d) - each token gets its frame's conditioning
        batch_idx = t.arange(b, device=x.device).view(b, 1).expand(b, l)  # (b, l)
        scale_exp = scale[batch_idx, frame_idx.unsqueeze(0).expand(b, l), :]  # (b, l, d)
        shift_exp = shift[batch_idx, frame_idx.unsqueeze(0).expand(b, l), :]
        gate_exp = gate[batch_idx, frame_idx.unsqueeze(0).expand(b, l), :]
        
        # Step 4: apply (x * scale + shift) * gate
        out = (x * scale_exp + shift_exp) * gate_exp
        return out


if __name__ == "__main__":
    import time
    
    # Setup
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")
    
    # Test parameters
    b = 32
    dur = 30
    toks_per_frame = 65
    d = 400
    l = dur * toks_per_frame
    
    # Create model and move to device
    model = CondBench(d).to(device)
    model.eval()
    
    # Create test data
    x = t.randn(b, l, d, device=device)
    cond_naive = t.randn(b, l, d, device=device)  # for naive method
    cond_compact = t.randn(b, dur, d, device=device)  # for reshape/scatter methods
    
    # Create compiled versions - use the same model instance!
    print("Compiling methods with torch.compile...")
    # Copy model state to ensure same weights
    model_compiled = CondBench(d).to(device)
    model_compiled.load_state_dict(model.state_dict())
    model_compiled.eval()
    
    # Compile each method from the same model instance
    naive_compiled = t.compile(model.naive, mode="reduce-overhead")
    reshape_compiled = t.compile(model.reshape_based, mode="reduce-overhead")
    scatter_compiled = t.compile(model.scatter_based, mode="reduce-overhead")
    
    # Warmup (including compilation)
    print("Warming up (including compilation)...")
    cond_expanded = cond_compact.repeat_interleave(toks_per_frame, dim=1)
    
    # Warmup non-compiled versions
    print("  Warming up non-compiled methods...")
    for _ in range(10):
        _ = model.naive(x, cond_expanded)
        _ = model.reshape_based(x, cond_compact, toks_per_frame)
        _ = model.scatter_based(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    
    # Warmup compiled versions - need more iterations for compilation and optimization
    print("  Warming up compiled methods (this may take a moment for compilation)...")
    # First call triggers compilation - do it separately
    _ = naive_compiled(x, cond_expanded)
    _ = reshape_compiled(x, cond_compact, toks_per_frame)
    _ = scatter_compiled(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    
    # Additional warmup iterations after compilation to ensure steady-state performance
    print("  Additional warmup iterations for compiled methods...")
    for _ in range(50):  # More iterations for compiled code to stabilize
        _ = naive_compiled(x, cond_expanded)
        _ = reshape_compiled(x, cond_compact, toks_per_frame)
        _ = scatter_compiled(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    print("  Warmup complete!")
    
    # Verify correctness
    print("\nVerifying correctness...")
    with t.no_grad():
        out_naive = model.naive(x, cond_expanded)
        out_reshape = model.reshape_based(x, cond_compact, toks_per_frame)
        out_scatter = model.scatter_based(x, cond_compact, toks_per_frame)
        out_naive_compiled = naive_compiled(x, cond_expanded)
        out_reshape_compiled = reshape_compiled(x, cond_compact, toks_per_frame)
        out_scatter_compiled = scatter_compiled(x, cond_compact, toks_per_frame)
        
        # Check if results match
        max_diff_reshape = (out_naive - out_reshape).abs().max().item()
        max_diff_scatter = (out_naive - out_scatter).abs().max().item()
        max_diff_naive_compiled = (out_naive - out_naive_compiled).abs().max().item()
        max_diff_reshape_compiled = (out_reshape - out_reshape_compiled).abs().max().item()
        max_diff_scatter_compiled = (out_scatter - out_scatter_compiled).abs().max().item()
        
        print(f"Max difference (naive vs reshape_based): {max_diff_reshape:.2e}")
        print(f"Max difference (naive vs scatter_based): {max_diff_scatter:.2e}")
        print(f"Max difference (naive vs naive_compiled): {max_diff_naive_compiled:.2e}")
        print(f"Max difference (reshape vs reshape_compiled): {max_diff_reshape_compiled:.2e}")
        print(f"Max difference (scatter vs scatter_compiled): {max_diff_scatter_compiled:.2e}")
        
        if (max_diff_reshape < 1e-5 and max_diff_scatter < 1e-5 and 
            max_diff_naive_compiled < 1e-5 and max_diff_reshape_compiled < 1e-5 and 
            max_diff_scatter_compiled < 1e-5):
            print("✓ All methods produce identical results!")
        else:
            print("✗ Results differ! Check implementation.")
    
    # Benchmark timing
    print("\nBenchmarking runtime...")
    n_iterations = 100
    
    # Time naive method (non-compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = model.naive(x, cond_expanded)
    if device == "cuda":
        t.cuda.synchronize()
    time_naive = (time.perf_counter() - start) / n_iterations
    
    # Time naive method (compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = naive_compiled(x, cond_expanded)
    if device == "cuda":
        t.cuda.synchronize()
    time_naive_compiled = (time.perf_counter() - start) / n_iterations
    
    # Time reshape_based method (non-compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = model.reshape_based(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    time_reshape = (time.perf_counter() - start) / n_iterations
    
    # Time reshape_based method (compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = reshape_compiled(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    time_reshape_compiled = (time.perf_counter() - start) / n_iterations
    
    # Time scatter_based method (non-compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = model.scatter_based(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    time_scatter = (time.perf_counter() - start) / n_iterations
    
    # Time scatter_based method (compiled)
    if device == "cuda":
        t.cuda.synchronize()
    start = time.perf_counter()
    with t.no_grad():
        for _ in range(n_iterations):
            _ = scatter_compiled(x, cond_compact, toks_per_frame)
    if device == "cuda":
        t.cuda.synchronize()
    time_scatter_compiled = (time.perf_counter() - start) / n_iterations
    
    # Print results
    print(f"\nAverage runtime over {n_iterations} iterations:")
    print(f"\n{'Method':<20} {'Non-compiled':<15} {'Compiled':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'naive':<20} {time_naive*1000:>8.3f} ms     {time_naive_compiled*1000:>8.3f} ms     {time_naive/time_naive_compiled:>6.2f}x")
    print(f"{'reshape_based':<20} {time_reshape*1000:>8.3f} ms     {time_reshape_compiled*1000:>8.3f} ms     {time_reshape/time_reshape_compiled:>6.2f}x")
    print(f"{'scatter_based':<20} {time_scatter*1000:>8.3f} ms     {time_scatter_compiled*1000:>8.3f} ms     {time_scatter/time_scatter_compiled:>6.2f}x")
    
    # Find fastest
    times = {
        "naive": time_naive,
        "naive_compiled": time_naive_compiled,
        "reshape_based": time_reshape,
        "reshape_based_compiled": time_reshape_compiled,
        "scatter_based": time_scatter,
        "scatter_based_compiled": time_scatter_compiled
    }
    fastest = min(times, key=times.get)
    print(f"\nFastest method: {fastest}")

