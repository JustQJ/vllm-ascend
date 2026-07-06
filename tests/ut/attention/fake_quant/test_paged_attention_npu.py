import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from tests.ut.attention.fake_quant.paged_attention_npu import paged_attention


NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
DTYPE = torch.bfloat16
KV_CACHE_CAPACITY_TOKENS = 16 * 1024
BLOCK_SIZE = 128
PREFILL_TARGETS = [8 * 1024]
DECODE_KV_TARGET = 8 * 1024
DECODE_Q_TARGETS = [1, 4]
BATCH_SIZES = [4]
BLOCK_SHAPES = [
    # (16, 32),
    # (16, 64),
    # (16, 128),
    # (16, 256),
    # (32, 32),
    # (32, 64),
    (32, 128),
    # (32, 256),
    # (64, 32),
    # (64, 64),
    # (64, 128),
    # (64, 256),
]


def _make_sparse_causal_mask(device):
    return torch.triu(torch.ones(2048, 2048), diagonal=1).to(
        torch.int8).to(device).contiguous()


def _varied_lengths(max_len, batch_size):
    if max_len == 1:
        return [1] * batch_size
    step = max(1, max_len // max(batch_size * 2, 1))
    return [max(1, max_len - seq_idx * step) for seq_idx in range(batch_size)]


def _scenario_cases():
    cases = []
    shape_idx = 0
    for batch_size in BATCH_SIZES:
        for target_len in PREFILL_TARGETS:
            block_m, block_n = BLOCK_SHAPES[shape_idx % len(BLOCK_SHAPES)]
            shape_idx += 1
            q_lens = _varied_lengths(target_len, batch_size)
            cases.append(pytest.param(
                "prefill",
                batch_size,
                q_lens,
                q_lens,
                block_m,
                block_n,
                id=f"prefill-bs{batch_size}-max{target_len}-bm{block_m}-bn{block_n}",
            ))

    for batch_size in BATCH_SIZES:
        kv_lens = _varied_lengths(DECODE_KV_TARGET, batch_size)
        for q_target in DECODE_Q_TARGETS:
            block_m, block_n = BLOCK_SHAPES[shape_idx % len(BLOCK_SHAPES)]
            shape_idx += 1
            q_lens = _varied_lengths(q_target, batch_size)
            cases.append(pytest.param(
                "decode_mtp",
                batch_size,
                q_lens,
                kv_lens,
                block_m,
                block_n,
                id=f"decode-bs{batch_size}-q{q_target}-kv8k-bm{block_m}-bn{block_n}",
            ))
    return cases


def _cumulative_lengths(lengths, device):
    return torch.tensor(_cumulative_lengths_list(lengths),
                        dtype=torch.int32).to(device)


def _cumulative_lengths_list(lengths):
    total = 0
    cumulative = []
    for length in lengths:
        total += int(length)
        cumulative.append(total)
    return cumulative


def _build_paged_inputs(q_lens, kv_lens, block_size, num_q_heads, num_kv_heads,
                        head_dim, dtype, device):
    batch_size = len(q_lens)
    blocks_per_seq = (KV_CACHE_CAPACITY_TOKENS + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    flat_kv_head_dim = num_kv_heads * head_dim

    query = torch.randn(sum(q_lens), num_q_heads, head_dim,
                        dtype=dtype) * 0.25
    key_cache = torch.zeros(num_blocks, block_size, flat_kv_head_dim,
                            dtype=dtype)
    value_cache = torch.zeros_like(key_cache)

    rows = []
    for seq_idx in range(batch_size):
        row = list(range(seq_idx * blocks_per_seq,
                         (seq_idx + 1) * blocks_per_seq))
        if seq_idx % 2 == 0:
            row.reverse()
        rows.append(row)
    block_table_cpu = torch.tensor(rows, dtype=torch.int32)

    for seq_idx, kv_len in enumerate(kv_lens):
        key_tokens = torch.randn(kv_len, num_kv_heads, head_dim,
                                 dtype=dtype) * 0.25
        value_tokens = torch.randn(kv_len, num_kv_heads, head_dim,
                                   dtype=dtype) * 0.25
        for token_idx in range(kv_len):
            logical_block = token_idx // block_size
            block_offset = token_idx % block_size
            physical_block = int(block_table_cpu[seq_idx, logical_block])
            key_cache[physical_block, block_offset] = key_tokens[token_idx].reshape(-1)
            value_cache[physical_block, block_offset] = value_tokens[token_idx].reshape(-1)

    actual_seq_qlen = _cumulative_lengths(q_lens, device)
    actual_seq_kvlen = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    sinks = (torch.randn(num_q_heads, dtype=torch.float32) * 0.1).to(
        dtype=dtype)
    return (
        query.to(device).contiguous(),
        key_cache.to(device).contiguous(),
        value_cache.to(device).contiguous(),
        block_table_cpu.to(device).contiguous(),
        actual_seq_qlen,
        actual_seq_kvlen,
        sinks.to(device).contiguous(),
    )


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_matches_fias_v2_with_qwen3_moe_scenarios(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    del scenario
    assert batch_size == len(q_lens)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for torch_npu FIAS v2 comparison")

    torch.manual_seed(0)
    device = "npu"
    softmax_scale = HEAD_DIM ** -0.5

    query, key_cache, value_cache, block_table, actual_seq_qlen, actual_seq_kvlen, sinks = (
        _build_paged_inputs(q_lens, kv_lens, BLOCK_SIZE, NUM_Q_HEADS,
                            NUM_KV_HEADS, HEAD_DIM, DTYPE, device)
    )
    actual_seq_qlen_list = _cumulative_lengths_list(q_lens)
    actual_seq_kvlen_list = [int(length) for length in kv_lens]

    if max(q_lens) == 1:
        sparse_mode = 0
        atten_mask = None
    else:
        sparse_mode = 3
        atten_mask = _make_sparse_causal_mask(device)

    triton_out = paged_attention(query, key_cache, value_cache, block_table,
                                 actual_seq_qlen, actual_seq_kvlen,
                                 NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
                                 BLOCK_SIZE, block_m, block_n, sinks,
                                 atten_mask)
    torch.npu.synchronize()

    fias_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query=query,
        key=key_cache,
        value=value_cache,
        num_query_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        input_layout="TND",
        pre_tokens=65535,
        next_tokens=0,
        atten_mask=atten_mask,
        sparse_mode=sparse_mode,
        softmax_scale=softmax_scale,
        block_table=block_table,
        block_size=BLOCK_SIZE,
        actual_seq_qlen=actual_seq_qlen_list,
        actual_seq_kvlen=actual_seq_kvlen_list,
        learnable_sink=sinks,
    )

    torch.testing.assert_close(triton_out, fias_out, atol=5e-2, rtol=5e-2)
