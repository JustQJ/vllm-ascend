import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from vllm_ascend.ops.triton.paged_attn.paged_attention_npu import (
    paged_attention as npu_paged_attention,
)
from tests.ut.attention.fake_quant.paged_attention_torch import (
    paged_attention as torch_paged_attention,
)
from tests.ut.attention.fake_quant.test_paged_attention_npu import (
    BLOCK_SIZE,
    DTYPE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    _build_paged_inputs,
    _cumulative_lengths_list,
    _make_sparse_causal_mask,
    _print_error_stats,
    _scenario_cases,
)


def _prepare_case(q_lens, kv_lens, device):
    softmax_scale = HEAD_DIM ** -0.5
    query, key_cache, value_cache, block_table, actual_seq_qlen, actual_seq_kvlen, sinks = (
        _build_paged_inputs(q_lens, kv_lens, BLOCK_SIZE, NUM_Q_HEADS,
                            NUM_KV_HEADS, HEAD_DIM, DTYPE, device)
    )

    if max(q_lens) == 1:
        sparse_mode = 0
        atten_mask = None
    else:
        sparse_mode = 3
        atten_mask = _make_sparse_causal_mask(device)

    return (
        softmax_scale,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
        sinks,
        sparse_mode,
        atten_mask,
    )


def _case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    return (
        f"{scenario}-bs{batch_size}-qmax{max(q_lens)}-"
        f"kvmax{max(kv_lens)}-bm{block_m}-bn{block_n}"
    )


def _assert_npu_available():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for PagedAttention comparison")


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_torch_matches_npu_without_mxfp4(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    del scenario
    assert batch_size == len(q_lens)
    _assert_npu_available()

    torch.manual_seed(0)
    device = "npu"
    (
        softmax_scale,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
        sinks,
        _,
        atten_mask,
    ) = _prepare_case(q_lens, kv_lens, device)

    torch_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n, sinks, atten_mask, False)
    npu_out = npu_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n, sinks, atten_mask, False)
    torch.npu.synchronize()

    torch.testing.assert_close(torch_out, npu_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_torch_matches_fias_v2_without_mxfp4(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    del scenario
    assert batch_size == len(q_lens)
    _assert_npu_available()

    torch.manual_seed(0)
    device = "npu"
    (
        softmax_scale,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
        sinks,
        sparse_mode,
        atten_mask,
    ) = _prepare_case(q_lens, kv_lens, device)

    torch_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n, sinks, atten_mask, False)
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
        actual_seq_qlen=_cumulative_lengths_list(q_lens),
        actual_seq_kvlen=[int(length) for length in kv_lens],
        learnable_sink=sinks,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(torch_out, fias_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_torch_matches_npu_with_mxfp4_error_distribution(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    assert batch_size == len(q_lens)
    _assert_npu_available()

    torch.manual_seed(0)
    device = "npu"
    (
        softmax_scale,
        query,
        key_cache,
        value_cache,
        block_table,
        actual_seq_qlen,
        actual_seq_kvlen,
        sinks,
        _,
        atten_mask,
    ) = _prepare_case(q_lens, kv_lens, device)

    torch_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n, sinks, atten_mask, True)
    npu_out = npu_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n, sinks, atten_mask, True)
    torch.npu.synchronize()

    torch_cpu = torch_out.to(torch.float32).cpu()
    npu_cpu = npu_out.to(torch.float32).cpu()
    abs_error = (torch_cpu - npu_cpu).abs()
    rel_error = abs_error / torch_cpu.abs().clamp_min(1e-6)
    assert torch.isfinite(abs_error).all()
    assert torch.isfinite(rel_error).all()

    _print_error_stats(
        _case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n),
        abs_error,
        rel_error,
    )
