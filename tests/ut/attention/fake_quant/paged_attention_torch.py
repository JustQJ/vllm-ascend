"""Torch reference for the fake-quant PagedAttention test kernel.

The public ``paged_attention`` function intentionally mirrors
``vllm_ascend.ops.triton.paged_attn.paged_attention_npu.paged_attention`` so
tests can swap the Triton implementation for this golden path without
reshaping inputs.  It reconstructs per-sequence K/V from the paged cache and
block table, or slices contiguous TND K/V when ``block_table`` is None, then
runs the same TND attention math as the local FIA golden tests.
"""

import torch


_MXFP4_EBITS = 2
_MXFP4_MBITS = 3
_MXFP4_EMAX = 2
_MXFP4_MAX_NORM = 6.0
_MXFP4_BLOCK_SIZE = 32
_MXFP4_MIN_EXP = 0.0
_MXFP4_SCALE_FACTOR = 2.0
_MXFP4_INV_SCALE_FACTOR = 0.5
_MXFP4_EPSILON = 1.17e-38


def _normalize_kv_cache(cache, block_size, num_kv_heads, head_dim):
    if cache.dim() == 3:
        expected = num_kv_heads * head_dim
        assert cache.shape[1] == block_size
        assert cache.shape[2] == expected
        return cache.view(cache.shape[0], block_size, num_kv_heads, head_dim)

    if cache.dim() == 4:
        if cache.shape[1] == block_size:
            assert cache.shape[2] == num_kv_heads
            assert cache.shape[3] == head_dim
            return cache

        if cache.shape[1] == num_kv_heads:
            assert cache.shape[2] == block_size
            assert cache.shape[3] == head_dim
            return cache.permute(0, 2, 1, 3).contiguous()

    raise AssertionError(
        "KV cache must be shaped as (blocks, block_size, Hkv * D), "
        "(blocks, block_size, Hkv, D), or (blocks, Hkv, block_size, D)"
    )


def _normalize_contiguous_kv(cache, num_kv_heads, head_dim):
    if cache.dim() == 3:
        assert cache.shape[1] == num_kv_heads
        assert cache.shape[2] == head_dim
        return cache

    if cache.dim() == 2:
        expected = num_kv_heads * head_dim
        assert cache.shape[1] == expected
        return cache.view(cache.shape[0], num_kv_heads, head_dim)

    raise AssertionError(
        "Contiguous KV must be shaped as (tokens, Hkv, D) or "
        "(tokens, Hkv * D) when block_table is None"
    )


def _with_leading_zero(cu_q_lens, num_seqs):
    if cu_q_lens.shape[0] == num_seqs:
        return torch.cat([cu_q_lens.new_zeros((1,)), cu_q_lens])
    assert cu_q_lens.shape[0] == num_seqs + 1
    return cu_q_lens


def _mxfp4_quantize_dequantize(x, qdim):
    """MXFP4-C7 fake quantization, matching the Triton helper on p."""
    original_shape = x.shape
    normalized_qdim = qdim if qdim >= 0 else x.ndim + qdim
    reduction_dim = normalized_qdim + 1
    x = x.unflatten(qdim, (-1, _MXFP4_BLOCK_SIZE))

    max_val = torch.amax(x.abs(), dim=reduction_dim, keepdim=True)
    shared_exp = torch.ceil(
        torch.log2(max_val.clamp(min=_MXFP4_EPSILON) / 7.0)
    )
    shared_exp = shared_exp.clamp(-127, 127)

    x = x * torch.exp2(-shared_exp)
    private_exp = torch.floor(
        torch.log2(x.abs().clamp(min=_MXFP4_EPSILON))
    ).clamp(min=_MXFP4_MIN_EXP)
    x = x * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR
    x = torch.sign(x) * torch.floor(x.abs() + 0.5)
    x = (
        x * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)
    ).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)
    x = x * torch.exp2(shared_exp)
    return x.reshape(original_shape)


def _mxfp4_p_tile(p, block_m, block_n):
    assert block_n % _MXFP4_BLOCK_SIZE == 0
    q_len = p.shape[-2]
    kv_len = p.shape[-1]
    pad_m = block_m - q_len
    pad_n = block_n - kv_len
    if pad_m or pad_n:
        p = torch.nn.functional.pad(p, (0, pad_n, 0, pad_m))
    quantized = _mxfp4_quantize_dequantize(p, qdim=-1)
    return quantized[..., :q_len, :kv_len]


def _gather_sequence_kv(cache, block_table_row, kv_len, block_size):
    num_blocks = (kv_len + block_size - 1) // block_size
    pieces = []
    for logical_block in range(num_blocks):
        physical_block = int(block_table_row[logical_block].detach().cpu().item())
        valid_len = min(block_size, kv_len - logical_block * block_size)
        pieces.append(cache[physical_block, :valid_len])
    if not pieces:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    return torch.cat(pieces, dim=0)


def _build_attention_mask(q_len, kv_len, atten_mask, device):
    context_len = max(kv_len - q_len, 0)
    q_abs_pos = context_len + torch.arange(q_len, device=device)
    kv_pos = torch.arange(kv_len, device=device)
    visible = kv_pos.unsqueeze(0) <= q_abs_pos.unsqueeze(1)

    if atten_mask is None:
        return visible

    mask_rows, mask_cols = atten_mask.shape
    mask_q = q_abs_pos.unsqueeze(1) - context_len
    mask_k = kv_pos.unsqueeze(0) - context_len
    index_valid = (
        (mask_q >= 0)
        & (mask_q < mask_rows)
        & (mask_k >= 0)
        & (mask_k < mask_cols)
    )
    if index_valid.any():
        safe_q = mask_q.clamp(0, max(mask_rows - 1, 0)).long()
        safe_k = mask_k.clamp(0, max(mask_cols - 1, 0)).long()
        mask_value = atten_mask[safe_q, safe_k]
        visible = visible & (~index_valid | (mask_value == 0))
    return visible


def _attention_one_head(query, key, value, visible, sm_scale, block_m, block_n,
                        sink, use_mxfp4_p, out_dtype):
    q_len = query.shape[0]
    head_dim = query.shape[-1]
    output = torch.empty((q_len, head_dim), dtype=torch.float32,
                         device=query.device)

    for q_start in range(0, q_len, block_m):
        q_end = min(q_start + block_m, q_len)
        q_blk = query[q_start:q_end]
        visible_blk = visible[q_start:q_end]
        q_blk_len = q_end - q_start

        if sink is None:
            m_i = torch.full((q_blk_len,), float("-inf"), device=query.device)
        else:
            m_i = sink.to(torch.float32).expand(q_blk_len).clone()
        l_i = torch.ones((q_blk_len,), dtype=torch.float32,
                         device=query.device)
        acc = torch.zeros((q_blk_len, head_dim), dtype=torch.float32,
                          device=query.device)

        for kv_start in range(0, key.shape[0], block_n):
            kv_end = min(kv_start + block_n, key.shape[0])
            qk = torch.matmul(q_blk, key[kv_start:kv_end].transpose(0, 1))
            qk = qk * sm_scale
            tile_visible = visible_blk[:, kv_start:kv_end]
            qk = qk.masked_fill(~tile_visible, -1.0e20)

            m_ij = torch.maximum(m_i, torch.amax(qk, dim=-1))
            p = torch.exp(qk - m_ij[:, None])
            p = torch.where(tile_visible, p, torch.zeros_like(p))
            if use_mxfp4_p:
                p = _mxfp4_p_tile(p, block_m, block_n).to(
                    out_dtype).to(torch.float32)

            l_ij = p.sum(dim=-1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + torch.matmul(
                p, value[kv_start:kv_end])
            m_i = m_ij

        output[q_start:q_end] = acc / l_i[:, None]

    return output


def paged_attention(q, k_cache, v_cache, block_table, cu_q_lens, kv_lens,
                    num_q_heads, num_kv_heads, sm_scale, block_size,
                    BLOCK_M=16, BLOCK_N=128, sinks=None, atten_mask=None,
                    use_mxfp4_p=False):
    head_dim = q.shape[-1]
    assert q.dim() == 3
    assert q.shape[1] == num_q_heads
    assert num_q_heads % num_kv_heads == 0
    assert BLOCK_M in {16, 32, 64}
    assert BLOCK_N in {32, 64, 128, 256}
    assert cu_q_lens.dtype == torch.int64
    assert kv_lens.dtype == torch.int64
    is_contiguous_kv = block_table is None
    if not is_contiguous_kv:
        assert block_table.dtype == torch.int32
    if sinks is not None:
        assert sinks.shape[0] == num_q_heads
    if atten_mask is not None:
        assert atten_mask.dim() == 2

    if is_contiguous_kv:
        k_cache = _normalize_contiguous_kv(k_cache, num_kv_heads, head_dim)
        v_cache = _normalize_contiguous_kv(v_cache, num_kv_heads, head_dim)
    else:
        k_cache = _normalize_kv_cache(k_cache, block_size, num_kv_heads, head_dim)
        v_cache = _normalize_kv_cache(v_cache, block_size, num_kv_heads, head_dim)

    num_seqs = kv_lens.shape[0]
    cu_q_lens = _with_leading_zero(cu_q_lens, num_seqs)
    cu_q_lens_cpu = cu_q_lens.detach().cpu().tolist()
    kv_lens_cpu = kv_lens.detach().cpu().tolist()
    cu_k_lens_cpu = [0]
    for kv_len in kv_lens_cpu:
        cu_k_lens_cpu.append(cu_k_lens_cpu[-1] + int(kv_len))
    assert cu_q_lens_cpu[0] == 0
    assert 0 <= cu_q_lens_cpu[-1] <= q.shape[0]

    num_kv_groups = num_q_heads // num_kv_heads
    out = torch.empty_like(q)

    for seq_idx in range(num_seqs):
        q_start = int(cu_q_lens_cpu[seq_idx])
        q_end = int(cu_q_lens_cpu[seq_idx + 1])
        q_len = q_end - q_start
        kv_len = int(kv_lens_cpu[seq_idx])
        assert q_len >= 0
        assert kv_len >= 0
        if q_len == 0:
            continue

        query = q[q_start:q_end].to(torch.float32)
        if is_contiguous_kv:
            kv_start = cu_k_lens_cpu[seq_idx]
            kv_end = cu_k_lens_cpu[seq_idx + 1]
            key = k_cache[kv_start:kv_end].to(torch.float32)
            value = v_cache[kv_start:kv_end].to(torch.float32)
        else:
            key = _gather_sequence_kv(
                k_cache, block_table[seq_idx], kv_len, block_size
            ).to(torch.float32)
            value = _gather_sequence_kv(
                v_cache, block_table[seq_idx], kv_len, block_size
            ).to(torch.float32)
        visible = _build_attention_mask(q_len, kv_len, atten_mask, q.device)

        for q_head_idx in range(num_q_heads):
            kv_head_idx = q_head_idx // num_kv_groups
            sink = None if sinks is None else sinks[q_head_idx]
            output = _attention_one_head(
                query[:, q_head_idx],
                key[:, kv_head_idx],
                value[:, kv_head_idx],
                visible,
                sm_scale,
                BLOCK_M,
                BLOCK_N,
                sink,
                use_mxfp4_p,
                q.dtype,
            )
            out[q_start:q_end, q_head_idx] = output.to(q.dtype)

    return out
