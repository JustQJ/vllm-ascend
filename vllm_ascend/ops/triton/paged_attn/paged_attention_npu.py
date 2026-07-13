"""Triton PagedAttention aligned with torch_npu FIAS v2 TND path.

This mirrors the paged ``npu_fused_infer_attention_score_v2`` call in
``vllm_ascend/attention/attention_v1.py``:

  q          : (num_q_tokens, num_q_heads, head_dim)
  k_cache    : (num_blocks, block_size, num_kv_heads * head_dim)
  v_cache    : (num_blocks, block_size, num_kv_heads * head_dim)
  block_table: (num_seqs, max_blocks_per_seq) int32
               optional; when None, key/value are contiguous TND tensors for
               first prefill (PrefillNoCache)
  cu_q_lens  : (num_seqs,) production actual_seq_qlen, or
               (num_seqs + 1,) cumulative q lengths with leading 0
  kv_lens    : (num_seqs,) actual kv length per sequence, int64
  q_block_seq/q_block_local: per-query-block sequence and local block ids
  atten_mask  : optional 2D causal mask, where non-zero entries are masked
  sinks       : optional (num_q_heads,) attention sink bias

The public wrapper also accepts the local test-friendly 4D cache layouts
``(num_blocks, block_size, num_kv_heads, head_dim)`` and
``(num_blocks, num_kv_heads, block_size, head_dim)``; both are normalized to
the production 3D cache layout before launching the kernel.  When
``block_table`` is None, K/V are expected in contiguous TND layout
``(num_kv_tokens, num_kv_heads, head_dim)``.
"""

import torch
import triton
import triton.language as tl

DEVICE = "npu"


@triton.jit
def clip(x, min_val, max_val):
    """Clip x to [min_val, max_val]."""
    return tl.minimum(tl.maximum(x, min_val), max_val)


@triton.jit
def to_mxfp4c7(tensor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, p_cx=7.0):
    """Quantize tensor to MXFP4-C7 (block-shared-scale MXFP4 with p_scale).

    MXFP4: 2 exponent bits + 1 sign bit + 2 mantissa bits (mbits=3 with
    implicit leading 1).  Shared scale is computed per 32-element sub-block and
    divided by *p_cx* before ceiling, which has the effect of attenuating
    small values (equivalent to ~p_scale ≈ 0.93 for p_cx=7).

    Reference: ``tests/ut/attention/fake_quant/test_npu_fused_infer_attention_triton.py``
    """
    FP32_EXPONENT_BIAS = 127.0
    FP32_MIN_NORMAL = tl.exp2(-FP32_EXPONENT_BIAS + 1)
    ebits, mbits = 2.0, 3.0
    emax = tl.exp2(ebits - 1)
    max_norm = tl.exp2(emax) * (tl.exp2(mbits - 1) - 1) / tl.exp2(mbits - 2)

    NUM_SUB_BLOCKS: tl.constexpr = BLOCK_N // 32
    tensor = tl.reshape(tensor, (BLOCK_M, NUM_SUB_BLOCKS, 32))

    shared_exp = tl.max(tl.abs(tensor), axis=-1, keep_dims=True)
    shared_exp = shared_exp / p_cx

    mask = (shared_exp == 0).to(shared_exp.dtype)
    M = shared_exp + FP32_MIN_NORMAL * mask
    shared_exp = tl.ceil(tl.log2(M))

    mask = (tensor > -FP32_EXPONENT_BIAS).to(tensor.dtype)
    tensor = tensor * mask

    scale_emax = tl.exp2(8.0 - 1.0) - 1
    shared_exp = tl.where(shared_exp > scale_emax, float('nan'), shared_exp)
    shared_exp = tl.where(shared_exp < -scale_emax, -scale_emax, shared_exp)

    tensor = tensor / (tl.exp2(shared_exp))
    mask = (tensor == 0).to(tensor.dtype)
    private_exp = tl.floor(tl.log2(tl.abs(tensor) + mask))

    min_exp = -(tl.exp2(ebits - 1)) + 2
    private_exp = tl.maximum(private_exp, min_exp)

    tensor = tensor / (tl.exp2(private_exp)) * (tl.exp2(mbits - 2))
    tensor_sign = (tensor > 0).to(tensor.dtype) - (tensor < 0).to(tensor.dtype)
    tensor = tensor_sign * tl.floor(tl.abs(tensor) + 0.5)
    tensor = tensor / (tl.exp2(mbits - 2)) * (tl.exp2(private_exp))

    tensor = clip(tensor, -max_norm, max_norm)
    tensor = tl.where(tensor == float('inf'), float('inf'), tensor)
    tensor = tl.where(tensor == -float('inf'), -float('inf'), tensor)
    tensor = tl.where(tensor == float('nan'), float('nan'), tensor)

    recovered_tensor = tensor * (tl.exp2(shared_exp))
    recovered_tensor = tl.reshape(recovered_tensor, (BLOCK_M, BLOCK_N))
    return recovered_tensor


@triton.jit
def to_mxfp4c7_p_only(p, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      p_cx: tl.constexpr = 7.0):
    FP32_MIN_NORMAL = tl.exp2(-126.0)
    NUM_SUB_BLOCKS: tl.constexpr = BLOCK_N // 32

    x = tl.reshape(p, (BLOCK_M, NUM_SUB_BLOCKS, 32))

    max_val = tl.max(x, axis=-1, keep_dims=True)
    max_val = tl.maximum(max_val, FP32_MIN_NORMAL)

    shared_exp = tl.ceil(tl.log2(max_val / p_cx))
    shared_exp = tl.minimum(tl.maximum(shared_exp, -127.0), 127.0)

    x = x * tl.exp2(-shared_exp)

    private_exp = tl.floor(tl.log2(tl.maximum(x, FP32_MIN_NORMAL)))
    private_exp = tl.maximum(private_exp, 0.0)

    x = x * tl.exp2(-private_exp) * 2.0
    x = tl.floor(x + 0.5)
    x = x * 0.5 * tl.exp2(private_exp)
    x = tl.minimum(x, 6.0)

    x = x * tl.exp2(shared_exp)
    return tl.reshape(x, (BLOCK_M, BLOCK_N))

@triton.jit
def _paged_attn_fwd_inner(
        acc, l_i, m_i, q,
        K_base, V_base,
        block_tables_ptr,
        BLOCK_SIZE: tl.constexpr,
        stride_k_blk, stride_k_slot, stride_k_flat: tl.constexpr,
        stride_v_blk, stride_v_slot, stride_v_flat: tl.constexpr,
        qk_scale,
        kv_head_idx,
        kv_start,
        q_abs_pos,
        q_mask,
        kv_seq_len,
        context_len,
        atten_mask_ptr,
        stride_mask_q,
        stride_mask_k,
        mask_rows,
        mask_cols,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_ATTEN_MASK: tl.constexpr,
        IS_CONTIGUOUS_KV: tl.constexpr,
        USE_MXFP4_P: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    num_tiles = (kv_seq_len + BLOCK_N - 1) // BLOCK_N

    for j in range(num_tiles):
        seq_offset = j * BLOCK_N + offs_n
        if IS_CONTIGUOUS_KV:
            k_token = kv_start + seq_offset
            k_offset = (
                k_token[None, :] * stride_k_blk
                + kv_head_idx * stride_k_slot
                + offs_d[:, None] * stride_k_flat
            )
        else:
            slot_in_block = seq_offset % BLOCK_SIZE
            logical_block = seq_offset // BLOCK_SIZE
            kv_lane_valid = seq_offset < kv_seq_len
            phys_block = tl.load(
                block_tables_ptr + logical_block,
                mask=kv_lane_valid,
                other=0,
            ).to(tl.int64)

            flat_head_offset = kv_head_idx * HEAD_DIM
            k_offset = (
                phys_block[None, :] * stride_k_blk
                + slot_in_block[None, :] * stride_k_slot
                + (flat_head_offset + offs_d[:, None]) * stride_k_flat
            )
        k = tl.load(
            K_base + k_offset,
            mask=seq_offset[None, :] < kv_seq_len,
            other=0.0,
        )

        qk = tl.dot(q, k) * qk_scale
        causal_mask = seq_offset[None, :] <= q_abs_pos[:, None]
        if HAS_ATTEN_MASK:
            mask_k = seq_offset[None, :] - context_len
            mask_q = q_abs_pos[:, None] - context_len
            mask_index_valid = (
                (mask_q >= 0)
                & (mask_q < mask_rows)
                & (mask_k >= 0)
                & (mask_k < mask_cols)
            )
            safe_mask_q = tl.where(mask_index_valid, mask_q, 0)
            safe_mask_k = tl.where(mask_index_valid, mask_k, 0)
            mask_offsets = safe_mask_q * stride_mask_q + safe_mask_k * stride_mask_k
            mask_value = tl.load(
                atten_mask_ptr + mask_offsets,
                mask=mask_index_valid,
                other=0,
            )
            causal_mask = causal_mask & (mask_value == 0)
        kv_mask = seq_offset[None, :] < kv_seq_len
        attn_valid = q_mask[:, None] & causal_mask & kv_mask
        qk_for_max = tl.where(attn_valid, qk, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(qk_for_max, axis=1))
        p_arg = tl.where(attn_valid, qk - m_ij[:, None], -80.0)
        p = tl.exp(p_arg)
        p = tl.where(attn_valid, p, 0.0)
        if USE_MXFP4_P:
            p = to_mxfp4c7(p, BLOCK_M, BLOCK_N).to(p.dtype)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        if IS_CONTIGUOUS_KV:
            v_token = kv_start + seq_offset
            v_offset = (
                v_token[:, None] * stride_v_blk
                + kv_head_idx * stride_v_slot
                + offs_d[None, :] * stride_v_flat
            )
        else:
            v_offset = (
                phys_block[:, None] * stride_v_blk
                + slot_in_block[:, None] * stride_v_slot
                + (flat_head_offset + offs_d[None, :]) * stride_v_flat
            )
        v = tl.load(
            V_base + v_offset,
            mask=seq_offset[:, None] < kv_seq_len,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _paged_attn_fwd(
        Q, K_cache, V_cache, Out, block_table_ptr, atten_mask_ptr,
        cu_q_lens_ptr, cu_k_lens_ptr, q_block_seq_ptr, q_block_local_ptr,
        kv_lens_ptr, sink_ptr,
        stride_q_tok, stride_q_head, stride_q_dim: tl.constexpr,
        stride_k_blk, stride_k_slot, stride_k_flat: tl.constexpr,
        stride_v_blk, stride_v_slot, stride_v_flat: tl.constexpr,
        stride_o_tok, stride_o_head, stride_o_dim: tl.constexpr,
        stride_mask_q,
        stride_mask_k,
        block_table_stride: tl.int64,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_groups: tl.constexpr,
        qk_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        mask_rows,
        mask_cols,
        HAS_SINKS: tl.constexpr,
        HAS_ATTEN_MASK: tl.constexpr,
        IS_CONTIGUOUS_KV: tl.constexpr,
        USE_MXFP4_P: tl.constexpr,
        IS_DECODE_ONLY: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // num_kv_groups

    if IS_DECODE_ONLY:
        seq = q_block_idx.to(tl.int64)
        q_block_local = 0
        q_start = seq
        q_end = seq + 1
    else:
        seq = tl.load(q_block_seq_ptr + q_block_idx).to(tl.int64)
        q_block_local = tl.load(q_block_local_ptr + q_block_idx).to(tl.int64)
        q_start = tl.load(cu_q_lens_ptr + seq).to(tl.int64)
        q_end = tl.load(cu_q_lens_ptr + seq + 1).to(tl.int64)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + seq).to(tl.int64)
    if IS_CONTIGUOUS_KV:
        kv_start = tl.load(cu_k_lens_ptr + seq).to(tl.int64)
    else:
        kv_start = tl.full((), 0, dtype=tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    q_idx = q_start + q_block_local * BLOCK_M + offs_m
    q_local = q_idx - q_start
    q_mask = q_idx < q_end
    q_idx_safe = tl.where(q_mask, q_idx, q_start)

    context_len = tl.maximum(kv_len - q_len, 0)
    q_abs_pos = context_len + q_local

    offs_d = tl.arange(0, HEAD_DIM)
    q_offsets = (
        q_idx_safe[:, None] * stride_q_tok
        + q_head_idx * stride_q_head
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0.0)

    if HAS_SINKS:
        sink = tl.load(sink_ptr + q_head_idx).to(tl.float32)
        m_i = sink + tl.zeros([BLOCK_M], dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    seq_block_table = block_table_ptr + seq * block_table_stride
    acc, l_i, m_i = _paged_attn_fwd_inner(
        acc, l_i, m_i, q,
        K_base=K_cache, V_base=V_cache,
        block_tables_ptr=seq_block_table,
        BLOCK_SIZE=BLOCK_SIZE,
        stride_k_blk=stride_k_blk,
        stride_k_slot=stride_k_slot,
        stride_k_flat=stride_k_flat,
        stride_v_blk=stride_v_blk,
        stride_v_slot=stride_v_slot,
        stride_v_flat=stride_v_flat,
        qk_scale=qk_scale,
        kv_head_idx=kv_head_idx,
        kv_start=kv_start,
        q_abs_pos=q_abs_pos,
        q_mask=q_mask,
        kv_seq_len=kv_len,
        context_len=context_len,
        atten_mask_ptr=atten_mask_ptr,
        stride_mask_q=stride_mask_q,
        stride_mask_k=stride_mask_k,
        mask_rows=mask_rows,
        mask_cols=mask_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        HAS_ATTEN_MASK=HAS_ATTEN_MASK,
        IS_CONTIGUOUS_KV=IS_CONTIGUOUS_KV,
        USE_MXFP4_P=USE_MXFP4_P,
    )
    empty_row = l_i == 0.0
    safe_l_i = tl.where(empty_row, 1.0, l_i)
    acc = acc / safe_l_i[:, None]
    # acc = acc / l_i[:, None]
    o_offsets = (
        q_idx_safe[:, None] * stride_o_tok
        + q_head_idx * stride_o_head
        + offs_d[None, :] * stride_o_dim
    )
    tl.store(Out + o_offsets, acc.to(Out.type.element_ty),
             mask=q_mask[:, None])


def _normalize_kv_cache(cache, block_size, num_kv_heads, head_dim):
    if cache.dim() == 3:
        expected = num_kv_heads * head_dim
        assert cache.shape[1] == block_size
        assert cache.shape[2] == expected
        return cache

    if cache.dim() == 4:
        if cache.shape[1] == block_size:
            assert cache.shape[2] == num_kv_heads
            assert cache.shape[3] == head_dim
            return cache.reshape(cache.shape[0], block_size,
                                 num_kv_heads * head_dim)

        if cache.shape[1] == num_kv_heads:
            assert cache.shape[2] == block_size
            assert cache.shape[3] == head_dim
            cache = cache.permute(0, 2, 1, 3).contiguous()
            return cache.reshape(cache.shape[0], block_size,
                                 num_kv_heads * head_dim)

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


class _paged_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, block_table,
                cu_q_lens, kv_lens,
                num_q_heads, num_kv_heads,
                sm_scale, block_size, BLOCK_M=16, BLOCK_N=64, sinks=None,
                atten_mask=None, use_mxfp4_p=False):
        del ctx
        head_dim = q.shape[-1]
        assert q.dim() == 3
        assert q.shape[1] == num_q_heads
        assert num_q_heads % num_kv_heads == 0
        # assert BLOCK_M in {16, 32, 64}
        # assert BLOCK_N in {32, 64, 128, 256}
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
            k_cache = _normalize_contiguous_kv(k_cache, num_kv_heads,
                                               head_dim)
            v_cache = _normalize_contiguous_kv(v_cache, num_kv_heads,
                                               head_dim)
        else:
            k_cache = _normalize_kv_cache(k_cache, block_size, num_kv_heads,
                                          head_dim)
            v_cache = _normalize_kv_cache(v_cache, block_size, num_kv_heads,
                                          head_dim)

        num_seqs = kv_lens.shape[0]
        if cu_q_lens.shape[0] == num_seqs:
            cu_q_lens = torch.cat([cu_q_lens.new_zeros((1,)), cu_q_lens])
        if is_contiguous_kv:
            cu_k_lens = torch.cat([
                kv_lens.new_zeros((1,)),
                torch.cumsum(kv_lens, dim=0, dtype=torch.int64),
            ])
        else:
            cu_k_lens = cu_q_lens
        # 每个 sequence 的 query 长度 & block 数
        seq_q_lens = cu_q_lens[1:] - cu_q_lens[:-1]          # [num_seqs]
        seq_q_blocks = (seq_q_lens + BLOCK_M - 1) // BLOCK_M  # [num_seqs], ceil 除法
        # q_block_seq：把 seq_idx 按 block 数 repeat
        q_block_seq = torch.repeat_interleave(
            torch.arange(num_seqs, dtype=torch.int64, device=q.device),
            seq_q_blocks.to(torch.int64),
        )  # [total_q_blocks]

        total_q_blocks = q_block_seq.shape[0]

        # q_block_local：每个 seq 内 0, 1, 2, ..., seq_q_blocks[i]-1
        # 构造 prefix starts：每个 seq 在展平数组里的起始位置
        cum_blocks = torch.cumsum(seq_q_blocks, dim=0)        # [num_seqs]
        seq_starts = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=q.device),
            cum_blocks[:-1],
        ])  # [num_seqs]，例如 seq_q_blocks=[3,2,4] → [0,3,5]
        seq_starts_expanded = torch.repeat_interleave(
            seq_starts, seq_q_blocks.to(torch.int64),
        )  # [total_q_blocks]
        q_block_local = (
            torch.arange(total_q_blocks, dtype=torch.int64, device=q.device)
            - seq_starts_expanded
        )

        qk_scale = sm_scale
        num_kv_groups = num_q_heads // num_kv_heads
        out = torch.empty_like(q)
        grid = (total_q_blocks, num_q_heads)
        mask_rows = 0
        mask_cols = 0
        stride_mask_q = 0
        stride_mask_k = 0
        atten_mask_ptr = q
        if atten_mask is not None:
            atten_mask_ptr = atten_mask
            mask_rows = atten_mask.shape[0]
            mask_cols = atten_mask.shape[1]
            stride_mask_q = atten_mask.stride(0)
            stride_mask_k = atten_mask.stride(1)

        num_programs = grid[0] * grid[1]
        assert num_programs <= 65535, (
            f"Ascend coreDim overflow: {num_programs} > 65535, "
            f"total_q_blocks={total_q_blocks}, "
            f"num_q_heads={num_q_heads}, "
            f"BLOCK_M={BLOCK_M}"
        )

        _paged_attn_fwd[grid](
            Q=q, K_cache=k_cache, V_cache=v_cache, Out=out,
            block_table_ptr=block_table if block_table is not None else q,
            atten_mask_ptr=atten_mask_ptr,
            cu_q_lens_ptr=cu_q_lens,
            cu_k_lens_ptr=cu_k_lens,
            q_block_seq_ptr=q_block_seq,
            q_block_local_ptr=q_block_local,
            kv_lens_ptr=kv_lens,
            sink_ptr=sinks,
            stride_q_tok=q.stride(0),
            stride_q_head=q.stride(1),
            stride_q_dim=q.stride(2),
            stride_k_blk=k_cache.stride(0),
            stride_k_slot=k_cache.stride(1),
            stride_k_flat=k_cache.stride(2),
            stride_v_blk=v_cache.stride(0),
            stride_v_slot=v_cache.stride(1),
            stride_v_flat=v_cache.stride(2),
            stride_o_tok=out.stride(0),
            stride_o_head=out.stride(1),
            stride_o_dim=out.stride(2),
            stride_mask_q=stride_mask_q,
            stride_mask_k=stride_mask_k,
            block_table_stride=block_table.stride(0) if block_table is not None else 0,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            num_q_heads=num_q_heads,
            num_kv_groups=num_kv_groups,
            qk_scale=qk_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            mask_rows=mask_rows,
            mask_cols=mask_cols,
            HAS_SINKS=sinks is not None,
            HAS_ATTEN_MASK=atten_mask is not None,
            IS_CONTIGUOUS_KV=is_contiguous_kv,
            USE_MXFP4_P=use_mxfp4_p,
            IS_DECODE_ONLY=False,
            num_warps=(4 if head_dim == 64 else 8),
        )
        return out

def paged_attention(
    query,
    key_cache,
    value_cache,
    block_table,
    actual_seq_qlen,
    actual_seq_kvlen,
    num_q_heads,
    num_kv_heads,
    softmax_scale,
    block_size,
    block_m=16,
    block_n=64,
    sinks=None,
    atten_mask=None,
    use_mxfp4_p=False
):
    return _paged_attention.apply(
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
        num_q_heads,
        num_kv_heads,
        softmax_scale,
        block_size,
        block_m,
        block_n,
        sinks,
        atten_mask,
        use_mxfp4_p
    )


def paged_attention_decode_out(
    query,
    key_cache,
    value_cache,
    block_table,
    actual_seq_qlen,
    actual_seq_kvlen,
    output,
    num_q_heads,
    num_kv_heads,
    softmax_scale,
    block_size,
    block_m=16,
    block_n=64,
    atten_mask=None,
    use_mxfp4_p=False,
):
    """Launch single-token paged decode into a caller-owned output tensor."""
    head_dim = query.shape[-1]
    assert query.dim() == 3
    assert query.shape == output.shape
    assert query.shape[1] == num_q_heads
    assert num_q_heads % num_kv_heads == 0
    assert block_table is not None and block_table.dtype == torch.int32
    assert actual_seq_kvlen.dtype in (torch.int32, torch.int64)
    assert actual_seq_kvlen.shape[0] >= query.shape[0]
    assert block_table.shape[0] >= query.shape[0]
    if atten_mask is not None:
        assert atten_mask.dim() == 2

    key_cache = _normalize_kv_cache(key_cache, block_size, num_kv_heads, head_dim)
    value_cache = _normalize_kv_cache(value_cache, block_size, num_kv_heads, head_dim)
    grid = (query.shape[0], num_q_heads)
    num_programs = grid[0] * grid[1]
    assert num_programs <= 65535, (
        f"Ascend coreDim overflow: {num_programs} > 65535, "
        f"batch_size={query.shape[0]}, num_q_heads={num_q_heads}"
    )

    mask_rows = 0
    mask_cols = 0
    stride_mask_q = 0
    stride_mask_k = 0
    atten_mask_ptr = query
    if atten_mask is not None:
        atten_mask_ptr = atten_mask
        mask_rows = atten_mask.shape[0]
        mask_cols = atten_mask.shape[1]
        stride_mask_q = atten_mask.stride(0)
        stride_mask_k = atten_mask.stride(1)

    _paged_attn_fwd[grid](
        Q=query,
        K_cache=key_cache,
        V_cache=value_cache,
        Out=output,
        block_table_ptr=block_table,
        atten_mask_ptr=atten_mask_ptr,
        cu_q_lens_ptr=query,
        cu_k_lens_ptr=query,
        q_block_seq_ptr=query,
        q_block_local_ptr=query,
        kv_lens_ptr=actual_seq_kvlen,
        sink_ptr=query,
        stride_q_tok=query.stride(0),
        stride_q_head=query.stride(1),
        stride_q_dim=query.stride(2),
        stride_k_blk=key_cache.stride(0),
        stride_k_slot=key_cache.stride(1),
        stride_k_flat=key_cache.stride(2),
        stride_v_blk=value_cache.stride(0),
        stride_v_slot=value_cache.stride(1),
        stride_v_flat=value_cache.stride(2),
        stride_o_tok=output.stride(0),
        stride_o_head=output.stride(1),
        stride_o_dim=output.stride(2),
        stride_mask_q=stride_mask_q,
        stride_mask_k=stride_mask_k,
        block_table_stride=block_table.stride(0),
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_q_heads=num_q_heads,
        num_kv_groups=num_q_heads // num_kv_heads,
        qk_scale=softmax_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        mask_rows=mask_rows,
        mask_cols=mask_cols,
        HAS_SINKS=False,
        HAS_ATTEN_MASK=atten_mask is not None,
        IS_CONTIGUOUS_KV=False,
        USE_MXFP4_P=use_mxfp4_p,
        IS_DECODE_ONLY=True,
        num_warps=(4 if head_dim == 64 else 8),
    )
    return output