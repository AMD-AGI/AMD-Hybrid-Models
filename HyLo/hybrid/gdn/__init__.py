# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# GDN (Gated Delta Network) layers for Zebra-Llama hybrid models.
# Uses fla (flash-linear-attention) from 3rdparty/flash-linear-attention.

from .hybrid_model import GDNDecoderLayer

__all__ = ["GDNDecoderLayer"]
