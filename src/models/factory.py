import os
import json

import torch as t 
from torch import nn

from ..models.dit import get_model as dit 
from ..models.scd import get_model as scd
from ..config import Config



def load_model_from_config(config_path: str, checkpoint_path: str = None, strict: bool = True) -> nn.Module:
    print(f"loading {config_path}")
    cmodel = Config.from_yaml(config_path).model
    ctrain = Config.from_yaml(config_path).train
    dtype = ctrain.dtype if "dtype" in ctrain else t.float32 
    if dtype == "bf16" or dtype == "bfloat16":
        dtype = t.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = t.float16
    
    if cmodel.model_id == "dit":
        model = dit(cmodel.height, cmodel.width, 
            n_window=cmodel.n_window, 
            patch_size=cmodel.patch_size, 
            n_heads=cmodel.n_heads, d_model=cmodel.d_model, 
            n_blocks=cmodel.n_blocks, 
            T=cmodel.T, 
            in_channels=cmodel.in_channels,
            bidirectional=cmodel.bidirectional,
            rope_type=cmodel.rope_type,
            C=cmodel.C,
            use_flex=cmodel.use_flex)
    elif cmodel.model_id == "scd":
        model = scd(cmodel.height, cmodel.width, 
            n_window=cmodel.n_window, 
            patch_size=cmodel.patch_size, 
            n_heads=cmodel.n_heads, d_model=cmodel.d_model, 
            n_blocks_encoder=cmodel.n_blocks_encoder, 
            n_blocks_decoder=cmodel.n_blocks_decoder,
            T=cmodel.T, 
            in_channels=cmodel.in_channels,
            bidirectional=cmodel.bidirectional,
            rope_type=cmodel.rope_type,
            C=cmodel.C,
            use_flex=cmodel.use_flex)
    else:
        raise ValueError(f"Invalid model type: {cmodel.model_id}")

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

    return model.to(dtype)
