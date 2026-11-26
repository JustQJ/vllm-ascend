# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP4 E2M1 RNE V quantization and addressed PA_BBND cache writes."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _encode_e2m1(x):
    a = tl.abs(x)
    # Alternating strict/inclusive midpoints implement ties-to-even.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    sign = (x.to(tl.int32, bitcast=True) >> 28) & 8
    return (code | sign).to(tl.uint8)


@triton.jit(do_not_specialize=["TOKENS"])
def _write_cache(
    key,
    value,
    key_scale,
    reciprocal,
    slots,
    key_cache,
    value_cache,
    scale_cache,
    TOKENS,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    K_PITCH: tl.constexpr,
    V_PITCH: tl.constexpr,
    S_PITCH: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    token, head = row // HEADS, row % HEADS
    valid = token < TOKENS
    slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(tl.int64)
    valid = valid & (slot >= 0)
    page, offset = tl.maximum(slot, 0) // BLOCK_TOKENS, tl.maximum(slot, 0) % BLOCK_TOKENS
    byte = tl.arange(0, DIM // 2)
    base = token.to(tl.int64) * V0 + head.to(tl.int64) * V1
    even = tl.load(value + base[:, None] + (2 * byte)[None, :] * V2, mask=valid[:, None], other=0).to(tl.float32)
    odd = tl.load(value + base[:, None] + (2 * byte + 1)[None, :] * V2, mask=valid[:, None], other=0).to(tl.float32)
    r0 = tl.load(reciprocal + head[:, None] * DIM + (2 * byte)[None, :]).to(tl.float32)
    r1 = tl.load(reciprocal + head[:, None] * DIM + (2 * byte + 1)[None, :]).to(tl.float32)
    packed = _encode_e2m1(even * r0) | (_encode_e2m1(odd * r1) << 4)
    payload_offset = (offset * HEADS + head).to(tl.int64)[:, None] * (DIM // 2) + byte[None, :]
    k = tl.load(key + row[:, None] * (DIM // 2) + byte[None, :], mask=valid[:, None], other=0)
    tl.store(key_cache + page[:, None] * K_PITCH + payload_offset, k, mask=valid[:, None])
    tl.store(value_cache + page[:, None] * V_PITCH + payload_offset, packed, mask=valid[:, None])
    group = tl.arange(0, DIM // 32)
    scales = tl.load(key_scale + row[:, None] * (DIM // 32) + group[None, :], mask=valid[:, None], other=0)
    scale_offset = (offset * HEADS + head).to(tl.int64)[:, None] * (DIM // 32) + group[None, :]
    tl.store(scale_cache + page[:, None] * S_PITCH + scale_offset, scales, mask=valid[:, None])


def write_mxfp4_cache(key, value, key_scale, reciprocal, kv_cache, slot_mapping):
    """Write only nonnegative slots; the first logical element occupies the low nibble.

    Quantized K and its scales come from npu_dynamic_mx_quant. V is read with
    its original strides, quantized in FP32 and packed directly into cache.
    All cache offsets use bytes, including a potentially padded page pitch.
    """
    tokens, heads, dim = value.shape
    if dim not in (128, 256):
        raise ValueError("MXFP4 writer requires D=128 or D=256")
    if tuple(key.shape) != (tokens, heads, dim // 2) or not key.is_contiguous():
        raise ValueError("MXFP4 K must be contiguous packed [T,H,D/2]")
    if tuple(key_scale.shape) != (tokens, heads, dim // 64, 2) or not key_scale.is_contiguous():
        raise ValueError("MXFP4 K scale must be contiguous [T,H,D/64,2]")
    if reciprocal.numel() != heads * dim or slot_mapping.numel() != tokens:
        raise ValueError("MXFP4 writer scale/slot shape mismatch")
    if not tokens:
        return
    k, v, ks, _ = kv_cache
    _write_cache[(tokens * heads,)](
        key.view(torch.uint8),
        value,
        key_scale.view(torch.uint8),
        reciprocal,
        slot_mapping,
        k,
        v,
        ks,
        tokens,
        heads,
        dim,
        *value.stride(),
        slot_mapping.stride(0),
        k.shape[1],
        k.stride(0),
        v.stride(0),
        ks.stride(0),
        ROWS=1,
    )
