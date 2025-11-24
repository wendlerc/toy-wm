import os
import re
import json
import time
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Dict, Any, List

import torch as t 
from torch import nn

from ..models.dit_dforce import get_model as dit_dforce 
from ..models.dit import get_model as dit
from ..config import Config

import yaml


def load_model_from_config(config_path: str, checkpoint_path: str = None, strict: bool = True) -> nn.Module:
    print(f"loading {config_path}")
    cmodel = Config.from_yaml(config_path).model
    
    if cmodel.model_id == "dit_dforce":
        get_model = dit_dforce
    elif cmodel.model_id == "dit":
        get_model = dit
    else:
        raise ValueError(f"Invalid model type: {cmodel.model_id}")
    C = cmodel.C if "C" in cmodel else 5000
    ln_first = cmodel.ln_first if "ln_first" in cmodel else False
    use_flex = cmodel.use_flex if "use_flex" in cmodel else False
    model = get_model(
        cmodel.height, cmodel.width, 
        n_window=cmodel.n_window, 
        patch_size=cmodel.patch_size, 
        n_heads=cmodel.n_heads, d_model=cmodel.d_model, 
        n_blocks=cmodel.n_blocks, 
        T=cmodel.T, 
        in_channels=cmodel.in_channels,
        bidirectional=cmodel.bidirectional,
        rope_type=cmodel.rope_type,
        C=cmodel.C,
        ln_first=ln_first,
        use_flex=use_flex
    )

    # If checkpoint_path is a folder, find top entry in ckpt_index.json
    if checkpoint_path is None and cmodel.checkpoint is not None:
        checkpoint_path = cmodel.checkpoint
    
    print(f"Loading checkpoint from {checkpoint_path}")

    if checkpoint_path is not None:
        if os.path.isdir(checkpoint_path):
            index_path = os.path.join(checkpoint_path, "ckpt_index.json")
            if not os.path.exists(index_path):
                raise ValueError(f"Directory '{checkpoint_path}' does not contain ckpt_index.json")
            with open(index_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            entries = d.get("entries", [])
            if not entries or not entries[0].get("path"):
                raise ValueError(f"No valid entries found in {index_path}")
            checkpoint_path = entries[0]["path"]

        state_dict = t.load(checkpoint_path, weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "_orig_mod." in list(state_dict.keys())[0]:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("_orig_mod.")}
        model.load_state_dict(state_dict, strict=strict)
        print('loaded state dict')
    return model

    

class CheckpointManager:
    """
    Manage top-K checkpoints by a metric. On each save:
      - Write a new checkpoint atomically
      - Keep only the top-K files by metric (max or min)
      - Delete files not in top-K
      - Maintain a small JSON index for quick reloads
    Also scans the directory on init to reconstruct state.

    Filenames are of the form: ckpt-step=<step>-metric=<metric>.pt
    """

    CKPT_PATTERN = re.compile(
        r"^ckpt-step=(?P<step>\d+)-metric=(?P<metric>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\.pt$"
    )

    def __init__(
        self,
        dirpath: str | Path,
        k: int = 5,
        mode: str = "max",  # or "min"
        metric_name: str = "score",
        is_main_process: bool = True,
        index_filename: str = "ckpt_index.json",
    ):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        assert mode in {"max", "min"}
        self.k = int(k)
        self.mode = mode
        self.metric_name = metric_name
        self.is_main = bool(is_main_process)
        self.index_path = self.dir / index_filename

        # entries: list of {path(str), step(int), metric(float), ts(float)}
        self.entries: List[Dict[str, Any]] = []

        self._load_index()
        self._scan_and_merge()
        self._prune_and_persist()

    # ---------- Public API ----------

    @property
    def best(self) -> Optional[Dict[str, Any]]:
        return self.entries[0] if self.entries else None

    @property
    def paths(self) -> List[str]:
        return [e["path"] for e in self.entries]

    @property
    def should_save(self) -> bool:
        """Use inside DDP loops to gate saving to rank-0 only."""
        return self.is_main

    def save(
        self,
        *,
        metric: float,
        step: int,
        model: Optional[nn.Module] = None,
        optimizer: Optional[t.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a checkpoint and keep only top-K by metric.

        Provide either `state_dict` or a `model` (optionally optimizer/scheduler).
        The saved file always contains:
           - 'model', 'optimizer', 'scheduler' (if provided)
           - 'step', metric_name, 'timestamp', 'manager'
        Returns info about the saved file and whether it made the top-K.
        """
        if not self.should_save:
            return {"saved": False, "kept": False, "reason": "not main process"}

        if state_dict is None:
            state_dict = {}
            if model is not None:
                state_dict["model"] = model.state_dict()
            if optimizer is not None:
                state_dict["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                # Some schedulers (e.g., OneCycleLR) have state_dict
                try:
                    state_dict["scheduler"] = scheduler.state_dict()
                except Exception:
                    pass

        ts = time.time()
        filename = f"ckpt-step={int(step):06d}-metric={float(metric):.8f}.pt"
        fpath = self.dir / filename

        # Attach metadata for convenience
        payload = {
            **state_dict,
            "step": int(step),
            self.metric_name: float(metric),
            "timestamp": ts,
            "manager": {
                "mode": self.mode,
                "k": self.k,
                "metric_name": self.metric_name,
                "filename": filename,
            },
        }

        # Atomic write
        with NamedTemporaryFile(dir=self.dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            t.save(payload, tmp_path)
            os.replace(tmp_path, fpath)  # atomic on POSIX
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        # Update entries and prune
        new_entry = {
            "path": str(fpath),
            "step": int(step),
            "metric": float(metric),
            "ts": ts,
        }
        self.entries.append(new_entry)
        kept = self._prune_and_persist()  # returns True if new file in top-K

        return {"saved": True, "kept": kept, "path": str(fpath), "best": self.best}

    # ---------- Internal helpers ----------

    def _sort_key(self, e: Dict[str, Any]):
        # For MAX: better first => sort by (-metric, step)
        # For MIN: better first => sort by (metric, step)
        return ((-e["metric"], e["step"]) if self.mode == "max" else (e["metric"], e["step"]))

    def _load_index(self):
        if not self.index_path.exists():
            self.entries = []
            return
        try:
            data = json.loads(self.index_path.read_text())
            entries = data.get("entries", [])
            # Drop missing files
            self.entries = [e for e in entries if Path(e["path"]).exists()]
            # Normalize types
            for e in self.entries:
                e["metric"] = float(e["metric"])
                e["step"] = int(e["step"])
                e["ts"] = float(e.get("ts", time.time()))
        except Exception:
            # If index is corrupted, fall back to empty and rescan
            self.entries = []

    def _scan_and_merge(self):
        """Scan directory for checkpoint files and merge with current entries."""
        seen = {Path(e["path"]).name for e in self.entries}
        for p in self.dir.glob("ckpt-step=*-metric=*.pt"):
            name = p.name
            if name in seen:
                continue
            m = self.CKPT_PATTERN.match(name)
            if not m:
                continue
            step = int(m.group("step"))
            try:
                metric = float(m.group("metric"))
            except ValueError:
                continue
            self.entries.append(
                {"path": str(p), "step": step, "metric": metric, "ts": p.stat().st_mtime}
            )

    def _prune_and_persist(self) -> bool:
        """Sort by metric, keep top-K, delete the rest. Return True if newest file is kept."""
        if not self.entries:
            self._persist_index()
            return False

        # Sort best-first
        self.entries.sort(key=self._sort_key)

        # Determine which to keep and which to delete
        keep = self.entries[: self.k]
        drop = self.entries[self.k :]

        keep_paths = {e["path"] for e in keep}
        newest_path = max(self.entries, key=lambda e: e["ts"])["path"]
        newest_kept = newest_path in keep_paths

        # Delete files not in top-K
        for e in drop:
            try:
                Path(e["path"]).unlink(missing_ok=True)
            except Exception:
                pass

        # Commit the top-K
        self.entries = keep
        self._persist_index()
        return newest_kept

    def _persist_index(self):
        data = {
            "k": self.k,
            "mode": self.mode,
            "metric_name": self.metric_name,
            "entries": self.entries,
            "updated_at": time.time(),
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.index_path)


# ---------------------- Example usage ----------------------
if __name__ == "__main__":
    # Example (single process). In DDP, construct with is_main_process=(rank==0).
    mgr = CheckpointManager("checkpoints", k=5, mode="max", metric_name="val_acc")

    model = nn.Linear(10, 2)
    opt = t.optim.AdamW(model.parameters(), lr=1e-3)

    # Fake loop
    for epoch in range(10):
        metric = 0.5 + 0.1 * t.rand(1).item()  # pretend validation accuracy
        info = mgr.save(metric=metric, step=epoch, model=model, optimizer=opt)
        print(
            f"epoch {epoch:02d} metric={metric:.4f} saved={info['saved']} kept={info['kept']} "
            f"best_metric={mgr.best['metric'] if mgr.best else None:.4f}"
        )

    print("Top-K paths:", mgr.paths)
    print("Best:", mgr.best)
