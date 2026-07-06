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
    PLOTDIR,
    _build_paged_inputs,
    _cumulative_lengths_list,
    _make_sparse_causal_mask,
    _print_error_stats,
    _save_error_distribution_plot,
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
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=False)
    npu_out = npu_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=False)
    torch.npu.synchronize()

    torch.testing.assert_close(torch_out, npu_out, atol=5e-2, rtol=5e-2)
    mse = (torch_out.float() - npu_out.float()).pow(2).mean().item()
    print(f"[torch-vs-npu-no-mxfp4] mse: {mse:.6e}")

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
    mse = (torch_out.float() - fias_out.float()).pow(2).mean().item()
    print(f"[torch-vs-fias_v2] mse: {mse:.6e}")

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
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=True)
    npu_out = npu_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=True)
    torch.npu.synchronize()

    torch_cpu = torch_out.to(torch.float32).cpu()
    npu_cpu = npu_out.to(torch.float32).cpu()
    abs_error = (torch_cpu - npu_cpu).abs()
    rel_error = abs_error / torch_cpu.abs().clamp_min(1e-6)
    assert torch.isfinite(abs_error).all()
    assert torch.isfinite(rel_error).all()

    mse = (torch_cpu - npu_cpu).pow(2).mean().item()
    _print_error_stats(
        _case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n),
        abs_error,
        rel_error,
        mse,
    )
    plot_path = f"{PLOTDIR}/{_case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n)}-torch-mxfp4-p-abs-error.png"
    _save_error_distribution_plot(
        abs_error,
        plot_path,
        f"Torch MXFP4_P vs NPU MXFP4_P abs error "
        f"({_case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n)})",
    )


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_torch_matches_fias_v1_with_qwen3_moe_scenarios(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    """对比 torch golden（无 sink）与 torch_npu.npu_fused_infer_attention_score（FIAS v1）。

    FIAS v1 不支持 learnable_sink，torch golden 和 FIAS v1 两侧均不传 sink。
    对应 production 中 DeviceOperator.npu_fused_infer_attention_score 路径。
    """
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
        _,          # sinks — FIAS v1 不支持，丢弃
        sparse_mode,
        atten_mask,
    ) = _prepare_case(q_lens, kv_lens, device)

    # ———— Torch golden（无 sink） ————
    torch_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,          # ← FIAS v1 无 sink
        atten_mask=atten_mask,
        use_mxfp4_p=False,
    )

    # ———— FIAS v1（对应 DeviceOperator → torch_npu.npu_fused_infer_attention_score） ————
    fias_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache,
        value=value_cache,
        num_heads=NUM_Q_HEADS,                # v1: num_heads
        num_key_value_heads=NUM_KV_HEADS,
        scale=softmax_scale,                   # v1: scale
        atten_mask=atten_mask,
        block_table=block_table,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=_cumulative_lengths_list(q_lens),       # v1: actual_seq_lengths
        actual_seq_lengths_kv=[int(length) for length in kv_lens],
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(torch_out, fias_out, atol=5e-2, rtol=5e-2)
    mse = (torch_out.float() - fias_out.float()).pow(2).mean().item()
    print(f"[torch-vs-fias_v1] mse: {mse:.6e}")


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_torch_mxfp4_p_self_comparison(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    """对比 torch_paged_attention 自身：开启 vs 不开启 MXFP4_P 的误差分布。"""
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
        _,          # sinks — 不使用
        _,
        atten_mask,
    ) = _prepare_case(q_lens, kv_lens, device)

    # ———— Torch golden：不开启 MXFP4_P ————
    base_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=False)

    # ———— Torch golden：开启 MXFP4_P ————
    mxfp4_p_out = torch_paged_attention(
        query, key_cache, value_cache, block_table, actual_seq_qlen,
        actual_seq_kvlen, NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None, atten_mask=atten_mask, use_mxfp4_p=True)
    torch.npu.synchronize()

    base_cpu = base_out.to(torch.float32).cpu()
    mxfp4_p_cpu = mxfp4_p_out.to(torch.float32).cpu()
    abs_error = (mxfp4_p_cpu - base_cpu).abs()
    rel_error = abs_error / base_cpu.abs().clamp_min(1e-6)
    mse = (mxfp4_p_cpu - base_cpu).pow(2).mean().item()

    assert torch.isfinite(abs_error).all()
    assert torch.isfinite(rel_error).all()
    assert abs_error.max().item() >= 0

    case_name = _case_name(scenario, batch_size, q_lens, kv_lens, block_m, block_n)
    _print_error_stats(case_name, abs_error, rel_error, mse)
    plot_path = f"{PLOTDIR}/{case_name}-torch-mxfp4-p-self-abs-error.png"
    _save_error_distribution_plot(
        abs_error,
        plot_path,
        f"Torch MXFP4_P ON vs OFF abs error ({case_name})",
    )
