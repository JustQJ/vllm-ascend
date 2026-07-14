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
