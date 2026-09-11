# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

import os

from safetensors import safe_open

def load_safetensors_to_dict(directory):
    safetensors_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    safetensors_dict[key] = f.get_tensor(key)
    return safetensors_dict

def construct_layer_dict(safetensors_dict, num_hidden_layers):
    layer_dict = {}
    non_mha_layer = [False for _ in range(num_hidden_layers)]
    prefix = "model.layers."
    for full_key, tensor in safetensors_dict.items():
        if full_key.startswith(prefix):
            parts = full_key[len(prefix):].split('.', 1)
            layer_id = int(parts[0])
            param_name = parts[1]
            if layer_id not in layer_dict:
                layer_dict[layer_id] = {}
            if "mla" in param_name:
                non_mha_layer[layer_id] = True
            layer_dict[layer_id][param_name] = tensor
    return layer_dict, non_mha_layer

def get_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        match = re.match(r"checkpoint-(\d+)", name)
        if match:
            step = int(match.group(1))
            checkpoint_dirs.append((step, os.path.join(output_dir, name)))

    if not checkpoint_dirs:
        return None

    # pick checkpoint with largest step number
    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs[-1][1]