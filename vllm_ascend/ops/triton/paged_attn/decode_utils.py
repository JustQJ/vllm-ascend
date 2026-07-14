DECODE_SPLIT_KV_NUM_PROGRAMS = 32
DECODE_SPLIT_KV_CHUNK_SIZES = (4096, 8192)


def build_split_kv_descriptors(
    kv_lens: list[int],
    block_size: int = 128,
    num_programs: int = DECODE_SPLIT_KV_NUM_PROGRAMS,
    chunk_sizes: tuple[int, ...] = DECODE_SPLIT_KV_CHUNK_SIZES,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Build fixed Split-KV block ranges for a small decode batch.

    Returns ``(work_desc, seq_desc, chunk_size)``. Each work descriptor is
    ``[seq_idx, logical_block_start, logical_block_end]``; unused work slots
    have ``seq_idx=-1``. Each sequence descriptor is
    ``[work_start, work_count]``.
    """
    if not kv_lens or len(kv_lens) not in (1, 2, 4):
        raise ValueError("Split-KV supports graph gears 1, 2, and 4")
    if block_size <= 0 or num_programs <= 0:
        raise ValueError("block_size and num_programs must be positive")
    if any(kv_len < 0 for kv_len in kv_lens):
        raise ValueError("kv_lens must be non-negative")

    selected_chunk_size = 0
    selected_counts: list[int] = []
    for chunk_size in chunk_sizes:
        if chunk_size <= 0 or chunk_size % block_size != 0:
            raise ValueError("chunk sizes must be positive multiples of block_size")
        counts = [
            (kv_len + chunk_size - 1) // chunk_size
            for kv_len in kv_lens
        ]
        if sum(counts) <= num_programs:
            selected_chunk_size = chunk_size
            selected_counts = counts
            break

    if selected_chunk_size == 0:
        raise ValueError(
            f"Split-KV requires more than {num_programs} programs: "
            f"kv_lens={kv_lens}, chunk_sizes={chunk_sizes}"
        )

    work_desc = [[-1, 0, 0] for _ in range(num_programs)]
    seq_desc: list[list[int]] = []
    blocks_per_chunk = selected_chunk_size // block_size
    work_start = 0
    for seq_idx, (kv_len, work_count) in enumerate(zip(kv_lens, selected_counts)):
        seq_desc.append([work_start, work_count])
        num_blocks = (kv_len + block_size - 1) // block_size
        for chunk_idx in range(work_count):
            block_start = chunk_idx * blocks_per_chunk
            block_end = min(block_start + blocks_per_chunk, num_blocks)
            work_desc[work_start + chunk_idx] = [
                seq_idx,
                block_start,
                block_end,
            ]
        work_start += work_count

    return work_desc, seq_desc, selected_chunk_size


def select_decode_heads_per_program(
    batch_size: int,
    num_q_heads: int,
    num_aicore: int,
) -> int:
    """Choose a power-of-two Q-head group for single-token decode."""
    assert batch_size > 0
    assert num_q_heads in (8, 16)
    assert num_aicore > 0

    target_groups = (num_aicore + batch_size - 1) // batch_size
    rounded_groups = 1 << (target_groups - 1).bit_length()
    num_head_groups = min(num_q_heads, rounded_groups)
    return num_q_heads // num_head_groups


def is_single_token_query(
    query_end_positions: object,
    batch_size: int,
) -> bool:
    """Return whether cumulative Q ends encode one token per sequence."""
    return isinstance(query_end_positions, list) and query_end_positions == list(range(1, batch_size + 1))


def supports_decode_specialization(
    *,
    enabled: bool,
    is_decode_only: bool,
    is_decoder_attention: bool,
    is_causal: bool,
    is_single_token_per_sequence: bool,
    has_sinks: bool,
    has_sliding_window: bool,
    has_speculative_config: bool,
    enable_c8_quant: bool,
    enable_hamming_sparse: bool,
    is_draft_model: bool,
    has_alibi: bool,
    has_logits_soft_cap: bool,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    query_ndim: int,
    query_is_bfloat16: bool,
    cache_is_bfloat16: bool,
    block_table_is_int32: bool,
) -> bool:
    """Return whether graph decode can use the specialized Triton kernel."""
    return (
        enabled
        and is_decode_only
        and is_decoder_attention
        and is_causal
        and is_single_token_per_sequence
        and not has_sinks
        and not has_sliding_window
        and not has_speculative_config
        and not enable_c8_quant
        and not enable_hamming_sparse
        and not is_draft_model
        and not has_alibi
        and not has_logits_soft_cap
        and num_q_heads in (8, 16)
        and num_kv_heads == 1
        and head_dim == 128
        and block_size == 128
        and query_ndim == 3
        and query_is_bfloat16
        and cache_is_bfloat16
        and block_table_is_int32
    )
