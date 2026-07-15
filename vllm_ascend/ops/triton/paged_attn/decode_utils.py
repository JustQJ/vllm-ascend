DECODE_SPLIT_KV_NUM_PROGRAMS = 32
DECODE_SPLIT_KV_REDUCE_NUM_PROGRAMS = 64
DECODE_SPLIT_KV_BATCH_SIZES = (1, 2, 4, 8)
DECODE_SPLIT_KV_MIN_CHUNK_SIZE = 2048
DECODE_SPLIT_KV_MAX_CHUNK_SIZE = 16384


def build_split_kv_descriptors(
    kv_lens: list[int],
    block_size: int = 128,
    num_programs: int = DECODE_SPLIT_KV_NUM_PROGRAMS,
    min_chunk_size: int = DECODE_SPLIT_KV_MIN_CHUNK_SIZE,
    max_chunk_size: int = DECODE_SPLIT_KV_MAX_CHUNK_SIZE,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Build balanced Split-KV block ranges for a small decode batch.

    Programs are assigned to non-empty sequences in proportion to their
    current per-program KV load. Each sequence is then divided into nearly
    equal contiguous block ranges. ``min_chunk_size`` prevents short
    sequences from being split too aggressively, while ``max_chunk_size``
    bounds the work assigned to any one program.

    Returns ``(work_desc, seq_desc, max_assigned_chunk_size)``. Each work
    descriptor is ``[seq_idx, logical_block_start, logical_block_end]``;
    unused work slots have ``seq_idx=-1``. Each sequence descriptor is
    ``[work_start, work_count]``.
    """
    if not kv_lens or len(kv_lens) not in DECODE_SPLIT_KV_BATCH_SIZES:
        raise ValueError("Split-KV supports graph gears 1, 2, 4, and 8")
    if block_size <= 0 or num_programs <= 0:
        raise ValueError("block_size and num_programs must be positive")
    if any(kv_len < 0 for kv_len in kv_lens):
        raise ValueError("kv_lens must be non-negative")
    if (
        min_chunk_size <= 0
        or max_chunk_size <= 0
        or min_chunk_size % block_size != 0
        or max_chunk_size % block_size != 0
        or min_chunk_size > max_chunk_size
    ):
        raise ValueError("chunk sizes must be ordered positive multiples of block_size")

    num_blocks = [(kv_len + block_size - 1) // block_size for kv_len in kv_lens]
    min_blocks_per_program = min_chunk_size // block_size
    max_blocks_per_program = max_chunk_size // block_size
    min_counts = [(blocks + max_blocks_per_program - 1) // max_blocks_per_program for blocks in num_blocks]
    max_counts = [(blocks + min_blocks_per_program - 1) // min_blocks_per_program for blocks in num_blocks]
    if sum(min_counts) > num_programs:
        raise ValueError(
            f"Split-KV requires more than {num_programs} programs: kv_lens={kv_lens}, max_chunk_size={max_chunk_size}"
        )

    work_counts = min_counts.copy()
    target_programs = min(num_programs, sum(max_counts))
    active_programs = sum(work_counts)
    while active_programs < target_programs:
        selected_seq = -1
        for seq_idx, blocks in enumerate(num_blocks):
            if work_counts[seq_idx] >= max_counts[seq_idx]:
                continue
            if selected_seq < 0:
                selected_seq = seq_idx
                continue
            if blocks * work_counts[selected_seq] > num_blocks[selected_seq] * work_counts[seq_idx]:
                selected_seq = seq_idx
        work_counts[selected_seq] += 1
        active_programs += 1

    work_desc = [[-1, 0, 0] for _ in range(num_programs)]
    seq_desc: list[list[int]] = []
    work_start = 0
    max_assigned_blocks = 0
    for seq_idx, (blocks, work_count) in enumerate(zip(num_blocks, work_counts)):
        seq_desc.append([work_start, work_count])
        if work_count:
            blocks_per_program, extra_blocks = divmod(blocks, work_count)
        else:
            blocks_per_program, extra_blocks = 0, 0
        block_start = 0
        for chunk_idx in range(work_count):
            assigned_blocks = blocks_per_program + (chunk_idx < extra_blocks)
            block_end = block_start + assigned_blocks
            work_desc[work_start + chunk_idx] = [
                seq_idx,
                block_start,
                block_end,
            ]
            max_assigned_blocks = max(max_assigned_blocks, assigned_blocks)
            block_start = block_end
        work_start += work_count

    return work_desc, seq_desc, max_assigned_blocks * block_size


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
