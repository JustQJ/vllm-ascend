# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static per-channel V and packed MXFP4 PA_BBND cache for QFA mode 5."""

import math
from dataclasses import replace

import torch
import torch_npu

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.attention.mxfp_rotation import rotate_heads
from vllm_ascend.attention.utils import is_dcp_enabled, is_pcp_enabled, notify_kv_cache_written
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start


def mxfp4_packet_section_sizes(heads, block_size, k_dim, v_dim):
    if heads <= 0 or block_size <= 0 or block_size % 64 or k_dim != v_dim or k_dim not in (128, 256):
        raise ValueError("MXFP4 requires H>0, S divisible by 64, and D=Dv in {128,256}")
    return (
        heads * block_size * k_dim // 2,
        heads * block_size * k_dim // 32,
        heads * block_size * v_dim // 2,
        heads * block_size * v_dim // 32,
    )


def mxfp4_cache_spec(spec):
    # Upstream hybrid alignment asks for a synthetic one-token page before
    # model loading. Physical cache views still require blocks divisible by 64.
    mxfp4_packet_section_sizes(
        spec.num_kv_heads, 64 if spec.block_size == 1 else spec.block_size, spec.head_size, spec.head_size_v
    )
    if spec.tokens_per_state != 1 or spec.num_heads != spec.num_kv_heads:
        raise ValueError("MXFP4 requires one state per token and KV head")
    width = spec.head_size + spec.head_size_v
    result = replace(spec, dtype=torch.uint8, state_content_bytes=width // 2 + width // 32)
    if result.page_size_padded is not None and result.page_size_padded < result.unpadded_page_size_bytes:
        raise ValueError("MXFP4 padded page is smaller than its payload")
    return result


def mxfp4_paged_cache_views(raw, blocks, heads, k_dim, v_dim, block_size, *, page_stride_bytes=None):
    """Return K,V,Kscale,Vscale byte views sharing [K|Ks|V|Vs|padding] pages."""
    if raw.ndim != 1 or raw.dtype != torch.int8 or raw.stride(0) != 1 or blocks <= 0:
        raise ValueError("MXFP4 needs a contiguous 1-D int8 allocation and positive block count")
    kb, ksb, vb, vsb = mxfp4_packet_section_sizes(heads, block_size, k_dim, v_dim)
    payload = kb + ksb + vb + vsb
    pitch = payload if page_stride_bytes is None else page_stride_bytes
    if pitch < payload or raw.numel() < blocks * pitch:
        raise ValueError("MXFP4 allocation/page pitch is smaller than the payload")
    shapes = (
        (blocks, block_size, heads, k_dim // 2),
        (blocks, block_size, heads, v_dim // 2),
        (blocks, block_size, heads, k_dim // 64, 2),
        (blocks, block_size // 64, heads, v_dim, 2),
    )
    offsets = (0, kb + ksb, kb, kb + ksb + vb)
    return tuple(
        torch.as_strided(
            raw.view(torch.uint8),
            shape,
            (pitch, *(math.prod(shape[i + 1 :]) for i in range(1, len(shape)))),
            storage_offset=raw.storage_offset() + offset,
        )
        for shape, offset in zip(shapes, offsets)
    )


def mxfp4_cache_views_for_spec(raw, spec, kernel_block_size):
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size:
        raise ValueError("MXFP4 scheduler pages must contain whole kernel blocks")
    ratio = spec.block_size // kernel_block_size
    if spec.page_size_bytes % ratio or raw.numel() % spec.page_size_bytes:
        raise ValueError("MXFP4 allocation/pitch must contain whole pages")
    return mxfp4_paged_cache_views(
        raw,
        raw.numel() // spec.page_size_bytes * ratio,
        spec.num_kv_heads,
        spec.head_size,
        spec.head_size_v,
        kernel_block_size,
        page_stride_bytes=spec.page_size_bytes // ratio,
    )


def fill_mxfp4_v_scale_cache(scale, cache):
    """Broadcast [H,D] over [P,S/64,H,D,2], also on recycled shared pages."""
    cache.copy_(scale.view(1, 1, cache.shape[2], cache.shape[3], 1))


class AscendC4MXFPAttentionBackend(AscendC8MXFPAttentionBackend):
    @classmethod
    def customize_spec(cls, spec):
        return mxfp4_cache_spec(spec)

    @staticmethod
    def get_impl_cls():
        if is_pcp_enabled() or is_dcp_enabled():
            raise NotImplementedError("MXFP4 QFA does not support PCP/DCP")
        return AscendC4MXFPAttentionBackendImpl


class AscendC4MXFPAttentionBackendImpl(AscendC8MXFPAttentionBackendImpl):
    k_quant_max = 7.0
    qfa_quant_mode = 5
    qfa_kv_layout = "PA_BBND"
    qfa_data_dtype = torch.float4_e2m1fn_x2
    qfa_allow_no_mask = False

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if output is None:
            raise ValueError("MXFP4 requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("MXFP4 output quantization is unsupported")
        if attn_metadata is None:
            return output.fill_(0)
        if self.vllm_config.kv_transfer_config is not None or getattr(self, "enable_hamming_sparse", False):
            raise NotImplementedError("MXFP4 currently requires local dense GQA cache")
        if kv_cache is None or len(kv_cache) != 4:
            raise ValueError("MXFP4 requires K,V,Kscale,Vscale cache views")
        record_attention_compute_start()
        count = attn_metadata.num_actual_tokens
        rotation = getattr(self, "_mxfp_rotation_mode", "none")
        query = query[:count]
        if rotation != "none":
            query = rotate_heads(query, layer._mxfp_rotation)
        q, qs = torch_npu.npu_dynamic_mx_quant(
            query, dst_type=torch_npu.float4_e2m1fn_x2, **getattr(self, "q_quant_options", {})
        )
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            from vllm_ascend.ops.triton.mxfp4_cache import write_mxfp4_cache
            from vllm_ascend.ops.triton.mxfp_v_scale import scatter_v_scales

            key, value = key[:count], value[:count]
            if rotation != "none":
                key = rotate_heads(key, layer._mxfp_rotation)
            if rotation == "qkv":
                value = rotate_heads(value, layer._mxfp_rotation)
            k, ks = torch_npu.npu_dynamic_mx_quant(
                key,
                dst_type=torch_npu.float4_e2m1fn_x2,
                block_size=32,
                scale_alg=2,
                dst_type_max=self.k_quant_max,
            )
            write_mxfp4_cache(
                k, value, ks, layer.v_cache_scale_float_reciprocal, kv_cache, attn_metadata.slot_mapping[:count]
            )
            scatter_v_scales(layer.v_cache_scale, kv_cache[3], attn_metadata.slot_mapping[:count])
            notify_kv_cache_written()
        output = self._forward_mxfp8_attention(q, qs, kv_cache, attn_metadata, output)
        if rotation == "qkv":
            # Restore channels before Qwen's elementwise output gate and o_proj.
            output[:count].copy_(rotate_heads(output[:count], layer._mxfp_inverse_rotation))
        return output
