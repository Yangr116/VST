# Copyright 2025 [Visual Spatial Tuning] Authors
# This file may have been modified by [Visual Spatial Tuning] Authors
# source file: https://github.com/ByteDance-Seed/VeOmni/blob/main/scripts/mereg_dcp_to_hf.py

import os
import argparse
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.models import build_foundation_model, build_processor, save_model_assets, save_model_weights


parser = argparse.ArgumentParser()
parser.add_argument("save_checkpoint_path", type=str)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--ckpt_manager", default='dcp')
args = parser.parse_args()


save_checkpoint_path = args.save_checkpoint_path
output_dir = str(Path(save_checkpoint_path).parent.parent) if args.output_dir is None else args.output_dir
ckpt_manager = args.ckpt_manager
os.makedirs(output_dir, exist_ok=True)
hf_weights_path = os.path.join(save_checkpoint_path, f"hf_ckpt_{str(Path(save_checkpoint_path).stem)}")

model_state_dict = ckpt_to_state_dict(
    save_checkpoint_path=save_checkpoint_path,
    output_dir=output_dir,
    ckpt_manager=ckpt_manager,
)

new_model_state_dict = dict()
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_key = k[len('module.'):]
    else:
        new_key = k
    new_model_state_dict[new_key] = v

asset_path = os.path.join(output_dir, 'model_assets')
model_assets = [AutoConfig.from_pretrained(asset_path), AutoProcessor.from_pretrained(asset_path)]
save_model_weights(hf_weights_path, new_model_state_dict, model_assets=model_assets)
