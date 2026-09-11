#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
import argparse
import os
# usage example: python convert_to_safetensors.py --checkpoint_dir checkpoints/<RUN_NAME>/checkpoint-XXXX/pytorch_model_fsdp_0 \
# --output_dir checkpoints/<RUN_NAME>/
def main():
    parser = argparse.ArgumentParser(
        description="Merge an Accelerate/Trainer FSDP sharded checkpoint into a single model.safetensors"
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to the checkpoint folder (e.g., .../checkpoint-38871)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to write model.safetensors",
    )
    args = parser.parse_args()

    ckpt = os.path.abspath(args.checkpoint_dir)
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)

    # Accelerate utility for SHARDED_STATE_DICT FSDP checkpoints
    try:
        from accelerate.utils import merge_fsdp_weights
    except ImportError:
        # older accelerate versions exposed it via fsdp_utils
        from accelerate.utils.fsdp_utils import merge_fsdp_weights

    merge_fsdp_weights(
        ckpt,
        out,
        safe_serialization=True,   # writes {output_dir}/model.safetensors
    )

    print(f"✅ Wrote: {os.path.join(out, 'model.safetensors')}")

if __name__ == "__main__":
    main()
