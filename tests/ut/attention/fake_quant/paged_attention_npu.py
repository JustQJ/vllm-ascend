"""Triton PagedAttention aligned with torch_npu FIAS v2 TND path.

This mirrors the paged ``npu_fused_infer_attention_score_v2`` call in
``vllm_ascend/attention/attention_v1.py``:

  q          : (num_q_tokens, num_q_heads, head_dim)
  k_cache    : (num_blocks, block_size, num_kv_heads * head_dim)
  v_cache    : (num_blocks, block_size, num_kv_heads * head_dim)
  block_table: (num_seqs, max_blocks_per_seq) int32
  cu_q_lens  : (num_seqs,) production actual_seq_qlen, or
               (num_seqs + 1,) cumulative q lengths with leading 0
  kv_lens    : (num_seqs,) actual kv length per sequence
  q_block_lens: (num_seqs + 1,) cumulative query-block lengths for BLOCK_M
  sinks       : optional (num_q_heads,) attention sink bias

The public wrapper also accepts the local test-friendly 4D cache layouts
``(num_blocks, block_size, num_kv_heads, head_dim)`` and
``(num_blocks, num_kv_heads, block_size, head_dim)``; both are normalized to
the production 3D cache layout before launching the kernel.
"""

import torch
import triton
import triton.language as tl

DEVICE = "npu"


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
        q_abs_pos,
        q_mask,
        kv_seq_len,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    num_tiles = (kv_seq_len + BLOCK_N - 1) // BLOCK_N

    for j in range(num_tiles):
        seq_offset = j * BLOCK_N + offs_n
        slot_in_block = seq_offset % BLOCK_SIZE
        logical_block = seq_offset // BLOCK_SIZE
        phys_block = tl.load(
            block_tables_ptr + logical_block
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
        kv_mask = seq_offset[None, :] < kv_seq_len
        qk = tl.where(q_mask[:, None] & causal_mask & kv_mask, qk, -1.0e20)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

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
        Q, K_cache, V_cache, Out, block_table_ptr,
        cu_q_lens_ptr, q_block_lens_ptr, kv_lens_ptr, sink_ptr,
        stride_q_tok, stride_q_head, stride_q_dim: tl.constexpr,
        stride_k_blk, stride_k_slot, stride_k_flat: tl.constexpr,
        stride_v_blk, stride_v_slot, stride_v_flat: tl.constexpr,
        stride_o_tok, stride_o_head, stride_o_dim: tl.constexpr,
        block_table_stride: tl.int64,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_groups: tl.constexpr,
        qk_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        num_seqs,
        HAS_SINKS: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // num_kv_groups

    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(q_block_lens_ptr + mid) <= q_block_idx:
            lo = mid + 1
        else:
            hi = mid
    seq = lo - 1
    if seq < 0:
        seq = 0

    q_start = tl.load(cu_q_lens_ptr + seq)
    q_end = tl.load(cu_q_lens_ptr + seq + 1)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + seq).to(tl.int32)

    q_block_start = tl.load(q_block_lens_ptr + seq)
    q_block_local = q_block_idx - q_block_start
    offs_m = tl.arange(0, BLOCK_M)
    q_idx = q_start + q_block_local * BLOCK_M + offs_m
    q_local = q_idx - q_start
    q_mask = q_idx < q_end

    context_len = tl.maximum(kv_len - q_len, 0)
    q_abs_pos = context_len + q_local

    offs_d = tl.arange(0, HEAD_DIM)
    q_offsets = (
        q_idx[:, None] * stride_q_tok
        + q_head_idx * stride_q_head
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0.0)

    if HAS_SINKS:
        sink = tl.load(sink_ptr + q_head_idx).to(tl.float32)
        m_i = sink + tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
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
        q_abs_pos=q_abs_pos,
        q_mask=q_mask,
        kv_seq_len=kv_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
    )

    acc = acc / l_i[:, None]
    o_offsets = (
        q_idx[:, None] * stride_o_tok
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


class _paged_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_cache, v_cache, block_table,
                cu_q_lens, kv_lens,
                num_q_heads, num_kv_heads,
                sm_scale, block_size, BLOCK_M=16, BLOCK_N=128, sinks=None):
        del ctx
        head_dim = q.shape[-1]
        assert q.dim() == 3
        assert q.shape[1] == num_q_heads
        assert num_q_heads % num_kv_heads == 0
        assert BLOCK_M in {16, 32, 64}
        assert BLOCK_N in {32, 64, 128, 256}
        if sinks is not None:
            assert sinks.shape[0] == num_q_heads

        k_cache = _normalize_kv_cache(k_cache, block_size, num_kv_heads,
                                      head_dim)
        v_cache = _normalize_kv_cache(v_cache, block_size, num_kv_heads,
                                      head_dim)

        num_seqs = kv_lens.shape[0]
        if cu_q_lens.shape[0] == num_seqs:
            cu_q_lens = torch.cat([cu_q_lens.new_zeros((1,)), cu_q_lens])
        assert cu_q_lens.shape[0] == num_seqs + 1
        cu_q_lens_cpu = cu_q_lens.detach().cpu().tolist()
        q_block_lens_list = [0]
        for seq_idx in range(num_seqs):
            seq_q_len = cu_q_lens_cpu[seq_idx + 1] - cu_q_lens_cpu[seq_idx]
            q_block_lens_list.append(q_block_lens_list[-1] +
                                     (seq_q_len + BLOCK_M - 1) // BLOCK_M)
        total_q_blocks = q_block_lens_list[-1]
        q_block_lens = torch.tensor(q_block_lens_list, dtype=torch.int32,
                                    device=q.device)

        qk_scale = sm_scale
        num_kv_groups = num_q_heads // num_kv_heads
        out = torch.empty_like(q)
        grid = (total_q_blocks, num_q_heads)

        _paged_attn_fwd[grid](
            Q=q, K_cache=k_cache, V_cache=v_cache, Out=out,
            block_table_ptr=block_table,
            cu_q_lens_ptr=cu_q_lens,
            q_block_lens_ptr=q_block_lens,
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
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            num_q_heads=num_q_heads,
            num_kv_groups=num_kv_groups,
            qk_scale=qk_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_seqs=num_seqs,
            HAS_SINKS=sinks is not None,
            num_warps=(4 if head_dim == 64 else 8),
        )
        return out


paged_attention = _paged_attention.apply
