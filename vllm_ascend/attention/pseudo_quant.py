# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-friendly pseudo-quantization helpers for attention experiments."""

import torch
import torch.nn.functional as F


def _pad_last_dim_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad the last dimension to a multiple of ``multiple``."""
    pad_amount = (multiple - x.size(-1) % multiple) % multiple
    return F.pad(x, (0, pad_amount))


def pseudo_quantize_fp4_per_block(
    x: torch.Tensor,
    block_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Simulate symmetric FP4 per-block quantization and dequantization."""
    original_shape = x.shape
    original_last_dim = x.size(-1)
    x_padded = _pad_last_dim_to_multiple(x, block_size)
    padded_last_dim = x_padded.size(-1)
    num_blocks = padded_last_dim // block_size
    x_blocks = x_padded.reshape(-1, num_blocks, block_size)

    fp4_max = 7.0
    scale = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp4_max
    scale = scale.to(scale_dtype).to(torch.float32)
    quantized = torch.clamp(
        torch.round(x_blocks.to(torch.float32) / scale),
        -fp4_max,
        fp4_max,
    )
    dequantized = (quantized * scale).to(x.dtype).reshape(x_padded.shape)
    return dequantized[..., :original_last_dim].reshape(original_shape)
