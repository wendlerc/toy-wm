"""Smoke tests: verify pong and doom training complete 2 steps with eval."""
import subprocess
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 600


def _run_training(config, extra_args=None):
    cmd = [sys.executable, "-m", "src.main", "--config", config]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "WANDB_MODE": "disabled"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=TIMEOUT,
        cwd=REPO_ROOT, env=env,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    return result


def test_pong_smoke():
    result = _run_training("configs/pong_test.yaml")
    assert result.returncode == 0, f"Pong training failed:\n{result.stderr[-1000:]}"
    assert "Model saved" in result.stdout, "Model was not saved"


def test_doom_smoke():
    result = _run_training(
        "configs/doom_test.yaml",
        ["--shard-dir", "/tmp/doom_latents"],
    )
    assert result.returncode == 0, f"Doom training failed:\n{result.stderr[-1000:]}"
    assert "Model saved" in result.stdout, "Model was not saved"
