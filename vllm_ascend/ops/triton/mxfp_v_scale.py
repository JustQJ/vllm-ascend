# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Write static V scales only to token groups owned by the current layer."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["TOKENS"])
def _scatter_v_scale_kernel(
    source,
    cache,
    slots,
    TOKENS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    DIM_GROUP_STRIDE: tl.constexpr,
    TOKEN_GROUP_STRIDE: tl.constexpr,
    DIM_INNER_STRIDE: tl.constexpr,
    PAIR_STRIDE: tl.constexpr,
    DIM_FRAGMENT: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.arange(0, CHANNELS)
    slot = tl.load(slots + token * SLOT_STRIDE).to(tl.int64)
    previous = tl.load(slots + (token - 1) * SLOT_STRIDE, mask=token > 0, other=-1).to(tl.int64)
    # Consecutive tokens in one 64-token scale pair need only one constant write.
    # A chunk beginning inside a group still writes its first token's scale.
    write = (token < TOKENS) & (slot >= 0) & ((token == 0) | (previous < 0) | (slot // 64 != previous // 64))
    safe_slot = tl.maximum(slot, 0)
    block, position = safe_slot // BLOCK_TOKENS, safe_slot % BLOCK_TOKENS
    head, dim = channel // DIM, channel % DIM
    offset = (
        block * BLOCK_STRIDE
        + head * HEAD_STRIDE
        + (dim // DIM_FRAGMENT) * DIM_GROUP_STRIDE
        + (position // 64) * TOKEN_GROUP_STRIDE
        + (dim % DIM_FRAGMENT) * DIM_INNER_STRIDE
    )
    value = tl.load(source + channel, mask=channel < HEADS * DIM, other=0)
    # QFA consumes both 32-token descales even when only half the pair is valid.
    # A stale 255 byte in the unused half otherwise propagates NaN through PV.
    tl.store(cache + offset, value, mask=write & (channel < HEADS * DIM))
    tl.store(cache + offset + PAIR_STRIDE, value, mask=write & (channel < HEADS * DIM))


def scatter_v_scales(value_scale: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Update both E8M0 descales per channel in each addressed 64-token pair.

    Handles C8 PA_NZ and C4 PA_BBND layouts, including padded physical pages.
    Negative padding slots never write. No host synchronization or dynamic
    tensor shapes are needed, so current slot mappings also work on graph replay.
    """
    if cache.ndim == 6:
        heads, dim = cache.shape[1], cache.shape[2] * cache.shape[4]
        block_tokens, fragment = cache.shape[3] * 64, cache.shape[4]
        pitch, head_stride, dim_group, token_group, dim_inner, pair = cache.stride()
    elif cache.ndim == 5:
        heads, dim = cache.shape[2], cache.shape[3]
        block_tokens, fragment = cache.shape[1] * 64, cache.shape[3]
        pitch, token_group, head_stride, dim_inner, pair = cache.stride()
        dim_group = 0
    else:
        raise ValueError("Static MXFP V scale cache requires PA_NZ or PA_BBND")
    if value_scale.numel() != heads * dim or not value_scale.is_contiguous():
        raise ValueError("Static MXFP V scales must be contiguous per-channel bytes")
    if value_scale.element_size() != 1 or cache.element_size() != 1 or slot_mapping.ndim != 1:
        raise ValueError("Static MXFP V scale cache requires byte scales and 1-D slots")
    if not slot_mapping.numel():
        return
    _scatter_v_scale_kernel[(slot_mapping.numel(),)](
        value_scale.view(torch.uint8),
        cache.view(torch.uint8),
        slot_mapping,
        slot_mapping.numel(),
        heads,
        dim,
        block_tokens,
        slot_mapping.stride(0),
        pitch,
        head_stride,
        dim_group,
        token_group,
        dim_inner,
        pair,
        fragment,
        CHANNELS=triton.next_power_of_2(heads * dim),
        num_warps=4,
    )
