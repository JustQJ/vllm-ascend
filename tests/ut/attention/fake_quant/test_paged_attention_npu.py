import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from vllm_ascend.ops.triton.paged_attn.paged_attention_npu import paged_attention


NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
DTYPE = torch.bfloat16
KV_CACHE_CAPACITY_TOKENS = 16 * 1024
BLOCK_SIZE = 128
PREFILL_TARGETS = [2*1024, 8*1024]
DECODE_KV_TARGET = 8 * 1024
DECODE_Q_TARGETS = [1, 4, 8]
BATCH_SIZES = [1, 2, 8, 16, 64, 128, 512]
BLOCK_SHAPES = [
    # (16, 32),
    # (16, 64),
    (16, 128),
    # (16, 256),
    # (32, 32),
    # (32, 64),
    # (32, 128),
    # (32, 256),
    # (64, 32),
    # (64, 64),
    # (64, 128),
    # (64, 256),
]
## TODO MODIFY this 
PLOTDIR = "/mnt/share/t00970481/mx-quant/figures1"


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
    cases.extend(_prefill_scenario_cases(shape_idx))
    shape_idx += len(cases)

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


def _prefill_scenario_cases(shape_idx=0):
    cases = []
    for batch_size in BATCH_SIZES:
        ## only when batch size is smaller than 8 for prefill scenario, otherwise the kernel will be error due to program is to larger
        if batch_size >= 4:
            continue
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
    return cases


def _cumulative_lengths(lengths, device):
    return torch.tensor(_cumulative_lengths_list(lengths),
                        dtype=torch.int64).to(device)


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
    actual_seq_kvlen = torch.tensor(kv_lens, dtype=torch.int64, device=device)
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


def _build_contiguous_inputs(q_lens, num_q_heads, num_kv_heads, head_dim,
                             dtype, device):
    query = torch.randn(sum(q_lens), num_q_heads, head_dim,
                        dtype=dtype) * 0.25
    key = torch.randn(sum(q_lens), num_kv_heads, head_dim,
                      dtype=dtype) * 0.25
    value = torch.randn_like(key)
    actual_seq_qlen = _cumulative_lengths(q_lens, device)
    actual_seq_kvlen = torch.tensor(q_lens, dtype=torch.int64, device=device)
    return (
        query.to(device).contiguous(),
        key.to(device).contiguous(),
        value.to(device).contiguous(),
        actual_seq_qlen,
        actual_seq_kvlen,
    )


def _safe_quantile(flat, quantiles, max_samples=200000):
    flat = flat.flatten()
    n = flat.numel()

    if n <= max_samples:
        return torch.quantile(flat, quantiles)

    gen = torch.Generator(device=flat.device)
    gen.manual_seed(0)

    idx = torch.randint(
        0,
        n,
        (max_samples,),
        device=flat.device,
        generator=gen,
    )
    sampled = flat[idx]
    return torch.quantile(sampled, quantiles)

def _print_error_stats(name, abs_error, rel_error, mse=None):
    abs_flat = abs_error.flatten()
    rel_flat = rel_error.flatten()
    quantiles = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
    abs_q = _safe_quantile(abs_flat, quantiles)
    rel_q = _safe_quantile(rel_flat, quantiles)
    print(
        f"\n[{name}] abs_error: "
        f"mean={abs_flat.mean().item():.6e}, "
        f"max={abs_flat.max().item():.6e}, "
        f"p50={abs_q[0].item():.6e}, "
        f"p90={abs_q[1].item():.6e}, "
        f"p99={abs_q[2].item():.6e}"
    )
    print(
        f"[{name}] rel_error: "
        f"mean={rel_flat.mean().item():.6e}, "
        f"max={rel_flat.max().item():.6e}, "
        f"p50={rel_q[0].item():.6e}, "
        f"p90={rel_q[1].item():.6e}, "
        f"p99={rel_q[2].item():.6e}"
    )
    if mse is not None:
        print(f"[{name}] mse: {mse:.6e}")


def _save_error_distribution_plot(abs_error, output_path, title):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pyplot = pytest.importorskip("matplotlib.pyplot")

    data = abs_error.flatten().numpy()
    pyplot.figure(figsize=(8, 5))
    pyplot.hist(data, bins=100, log=True)
    pyplot.xlabel("absolute error")
    pyplot.ylabel("count (log scale)")
    pyplot.title(title)
    pyplot.grid(True, alpha=0.3)
    pyplot.tight_layout()
    pyplot.savefig(output_path)
    pyplot.close()
    print(f"[MXFP4_P] abs error distribution plot: {output_path}")


def check_nan_inf(tensor, name):
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan_cnt = nan_mask.sum().item()
    inf_cnt = inf_mask.sum().item()
    total_elem = tensor.numel()
    
    print(f"\n==== {name} ====")
    print(f"total num: {total_elem}")
    print(f"NaN: {nan_cnt}, 占比: {nan_cnt / total_elem:.6f}")
    print(f"Inf: {inf_cnt}, 占比: {inf_cnt / total_elem:.6f}")
    if nan_cnt > 0:
        print("NaN")
    if inf_cnt > 0:
        print("Inf")
    return nan_cnt, inf_cnt

@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_mxfp4_p_error_distribution(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    assert batch_size == len(q_lens)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for Triton PagedAttention comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    softmax_scale = HEAD_DIM ** -0.5
    query, key_cache, value_cache, block_table, actual_seq_qlen, actual_seq_kvlen, _ = (
        _build_paged_inputs(q_lens, kv_lens, BLOCK_SIZE, NUM_Q_HEADS,
                            NUM_KV_HEADS, HEAD_DIM, DTYPE, device)
    )
    if max(q_lens) == 1:
        atten_mask = None
    else:
        atten_mask = _make_sparse_causal_mask(device)

    base_out = paged_attention(query, key_cache, value_cache, block_table,
                               actual_seq_qlen, actual_seq_kvlen,
                               NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
                               BLOCK_SIZE, block_m, block_n,
                               sinks=None, atten_mask=atten_mask,
                               use_mxfp4_p=False)
    mxfp4_p_out = paged_attention(query, key_cache, value_cache, block_table,
                                  actual_seq_qlen, actual_seq_kvlen,
                                  NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
                                  BLOCK_SIZE, block_m, block_n,
                                  sinks=None, atten_mask=atten_mask,
                                  use_mxfp4_p=True)
    torch.npu.synchronize()

    base_cpu = base_out.to(torch.float32).cpu()
    mxfp4_p_cpu = mxfp4_p_out.to(torch.float32).cpu()
    abs_error = (mxfp4_p_cpu - base_cpu).abs()
    rel_error = abs_error / base_cpu.abs().clamp_min(1e-6)

    mse = (mxfp4_p_cpu - base_cpu).pow(2).mean().item()
    assert torch.isfinite(rel_error).all()
    assert abs_error.max().item() >= 0

    case_name = (
        f"{scenario}-bs{batch_size}-qmax{max(q_lens)}-"
        f"kvmax{max(kv_lens)}-bm{block_m}-bn{block_n}"
    )
    _print_error_stats(case_name, abs_error, rel_error, mse)
    plot_path = f"{PLOTDIR}/{case_name}-mxfp4-p-abs-error.png"
    _save_error_distribution_plot(
        abs_error,
        plot_path,
        f"USE_MXFP4_P vs baseline abs error ({case_name})",
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
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
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
    # fias_out = triton_out
    # base_cpu = triton_out.to(torch.float32).cpu()
    # check_nan_inf(base_cpu, "triton_out")

    # base_cpu_nan_mask = torch.isnan(base_cpu).nonzero()
    # print(base_cpu_nan_mask[:50].tolist())
    # print("mxfp4_p_cpu shape", base_cpu.shape)

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

    mse = (triton_out.float() - fias_out.float()).pow(2).mean().item()
    print(f"[triton-vs-fias_v2] mse: {mse:.6e}")

    torch.testing.assert_close(triton_out, fias_out, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _scenario_cases(),
)
def test_paged_attention_matches_fias_v1_with_qwen3_moe_scenarios(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    """对比 Triton paged_attention（无 sink）与 torch_npu.npu_fused_infer_attention_score（FIAS v1）。

    FIAS v1 对应 DeviceOperator.npu_fused_infer_attention_score 路径：
    production 中默认 sparse_mode=3、无 sliding_window、无 learnable_sink 时走此分支。
    """
    del scenario
    assert batch_size == len(q_lens)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for torch_npu FIAS v1 comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    softmax_scale = HEAD_DIM ** -0.5

    # FIAS v1 不支持 learnable_sink，paged_attention 侧也传 sinks=None
    query, key_cache, value_cache, block_table, actual_seq_qlen, actual_seq_kvlen, _ = (
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

    # ———— Triton PagedAttention（无 sink） ————
    triton_out = paged_attention(
        query, key_cache, value_cache, block_table,
        actual_seq_qlen, actual_seq_kvlen,
        NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,          # ← FIAS v1 无 sink
        atten_mask=atten_mask,
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
        actual_seq_lengths=actual_seq_qlen_list,       # v1: actual_seq_lengths
        actual_seq_lengths_kv=actual_seq_kvlen_list,
        sparse_mode=sparse_mode,
    )

    mse = (triton_out.float() - fias_out.float()).pow(2).mean().item()
    print(f"[triton-vs-fias_v1] mse: {mse:.6e}")

    torch.testing.assert_close(triton_out, fias_out, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _prefill_scenario_cases(),
)
def test_paged_attention_matches_fias_v1_with_contiguous_prefill(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    assert scenario == "prefill"
    assert batch_size == len(q_lens)
    assert q_lens == kv_lens
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for torch_npu FIAS v1 comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    softmax_scale = HEAD_DIM ** -0.5

    query, key, value, actual_seq_qlen, actual_seq_kvlen = (
        _build_contiguous_inputs(q_lens, NUM_Q_HEADS, NUM_KV_HEADS,
                                 HEAD_DIM, DTYPE, device)
    )
    atten_mask = _make_sparse_causal_mask(device)

    triton_out = paged_attention(
        query, key, value, None,
        actual_seq_qlen, actual_seq_kvlen,
        NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,
        atten_mask=atten_mask,
    )

    fias_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        num_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        scale=softmax_scale,
        atten_mask=atten_mask,
        block_table=None,
        input_layout="TND",
        block_size=BLOCK_SIZE,
        actual_seq_lengths=_cumulative_lengths_list(q_lens),
        actual_seq_lengths_kv=_cumulative_lengths_list(kv_lens),
        sparse_mode=3,
    )
    torch.npu.synchronize()

    mse = (triton_out.float() - fias_out.float()).pow(2).mean().item()
    print(f"[triton-contiguous-prefill-vs-fias_v1] mse: {mse:.6e}")
    torch.testing.assert_close(triton_out, fias_out, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "scenario,batch_size,q_lens,kv_lens,block_m,block_n",
    _prefill_scenario_cases(),
)
def test_paged_attention_contiguous_prefill_mxfp4_p_error_distribution(
        scenario, batch_size, q_lens, kv_lens, block_m, block_n):
    assert scenario == "prefill"
    assert batch_size == len(q_lens)
    assert q_lens == kv_lens
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for Triton PagedAttention comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    softmax_scale = HEAD_DIM ** -0.5

    query, key, value, actual_seq_qlen, actual_seq_kvlen = (
        _build_contiguous_inputs(q_lens, NUM_Q_HEADS, NUM_KV_HEADS,
                                 HEAD_DIM, DTYPE, device)
    )
    atten_mask = _make_sparse_causal_mask(device)

    base_out = paged_attention(
        query, key, value, None,
        actual_seq_qlen, actual_seq_kvlen,
        NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,
        atten_mask=atten_mask,
        use_mxfp4_p=False,
    )
    mxfp4_p_out = paged_attention(
        query, key, value, None,
        actual_seq_qlen, actual_seq_kvlen,
        NUM_Q_HEADS, NUM_KV_HEADS, softmax_scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,
        atten_mask=atten_mask,
        use_mxfp4_p=True,
    )
    torch.npu.synchronize()

    base_cpu = base_out.to(torch.float32).cpu()
    mxfp4_p_cpu = mxfp4_p_out.to(torch.float32).cpu()
    abs_error = (mxfp4_p_cpu - base_cpu).abs()
    rel_error = abs_error / base_cpu.abs().clamp_min(1e-6)
    mse = (mxfp4_p_cpu - base_cpu).pow(2).mean().item()

    assert torch.isfinite(rel_error).all()
    assert abs_error.max().item() >= 0

    case_name = (
        f"contiguous-{scenario}-bs{batch_size}-qmax{max(q_lens)}-"
        f"kvmax{max(kv_lens)}-bm{block_m}-bn{block_n}"
    )
    _print_error_stats(case_name, abs_error, rel_error, mse)
    plot_path = f"{PLOTDIR}/{case_name}-mxfp4-p-abs-error.png"
    _save_error_distribution_plot(
        abs_error,
        plot_path,
        f"Contiguous prefill USE_MXFP4_P vs baseline abs error ({case_name})",
    )


def test_paged_attention_matches_fias_v1_with_contiguous_prefill1():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for torch_npu FIAS v1 comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    q_lens = [64, 48]
    block_m, block_n = BLOCK_SHAPES[0]
    softmax_scale = HEAD_DIM ** -0.5

    # query, key, value, actual_seq_qlen, actual_seq_kvlen = (
    #     _build_contiguous_inputs(q_lens, NUM_Q_HEADS, NUM_KV_HEADS,
    #                              HEAD_DIM, DTYPE, device)
    # )
    # atten_mask = _make_sparse_causal_mask(device)
    filepath = "/mnt/share/t00970481/dump_data_prefill/fused_attention_step_0.pt"
    data = torch.load(filepath)

    query, key, value, atten_mask, block_table, input_layout, block_size, actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, num_heads, scale, sparse_mode, key_cache, attn_output = data['query'].to(device), data['key'].to(device), data['value'].to(device), data['atten_mask'].to(device), data['block_table'], data['input_layout'], data['block_size'], data['actual_seq_lengths'], data['actual_seq_lengths_kv'], data['num_key_value_heads'], data['num_heads'], data['scale'], data['sparse_mode'], data['key_cache'], data['attn_output'].to(device)
    actual_seq_lengths1 = torch.tensor(actual_seq_lengths).to(device)
    actual_seq_lengths_kv1 = torch.tensor(actual_seq_lengths_kv).to(device)
    print(actual_seq_lengths1.dtype, actual_seq_lengths_kv1.dtype)
    triton_out = paged_attention(
        query, key, value, block_table,
        actual_seq_lengths1, actual_seq_lengths_kv1,
        num_heads, num_key_value_heads, scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,
        atten_mask=atten_mask,
    )

    fias_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale=scale,
        atten_mask=atten_mask,
        block_table=block_table,
        input_layout=input_layout,
        block_size=block_size,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()

    mse1 = (triton_out.float() - fias_out.float()).pow(2).mean().item()
    mse2 = (triton_out.float() - attn_output.float()).pow(2).mean().item()
    mse3 = (fias_out.float() - attn_output.float()).pow(2).mean().item()
    print(f"[triton-contiguous-prefill-vs-fias_v1] mse1: {mse1}, mse2: {mse2}, mse3: {mse3}")
    torch.testing.assert_close(triton_out, fias_out, atol=5e-3, rtol=5e-3)
    
    torch.testing.assert_close(attn_output, fias_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(triton_out, attn_output, atol=5e-3, rtol=5e-3)


def test_paged_attention_matches_fias_v1_with_decode1():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for torch_npu FIAS v1 comparison")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    device = "npu"
    q_lens = [64, 48]
    block_m, block_n = BLOCK_SHAPES[0]
    softmax_scale = HEAD_DIM ** -0.5

    # query, key, value, actual_seq_qlen, actual_seq_kvlen = (
    #     _build_contiguous_inputs(q_lens, NUM_Q_HEADS, NUM_KV_HEADS,
    #                              HEAD_DIM, DTYPE, device)
    # )
    # atten_mask = _make_sparse_causal_mask(device)
    filepath = "/mnt/share/t00970481/dump_data_decode/fused_attention_step_0.pt"
    data = torch.load(filepath)

    query, key, value, atten_mask, block_table, input_layout, block_size, actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, num_heads, scale, sparse_mode, key_cache, attn_output = data['query'].to(device), data['key'].to(device), data['value'].to(device), data['atten_mask'].to(device), data['block_table'].to(device), data['input_layout'], data['block_size'], data['actual_seq_lengths'], data['actual_seq_lengths_kv'], data['num_key_value_heads'], data['num_heads'], data['scale'], data['sparse_mode'], data['key_cache'], data['attn_output'].to(device)
    actual_seq_lengths1 = torch.tensor(actual_seq_lengths).to(device)
    actual_seq_lengths_kv1 = torch.tensor(actual_seq_lengths_kv).to(device)
    print(actual_seq_lengths1.dtype, actual_seq_lengths_kv1.dtype, block_table.dtype, query.dtype, value.dtype)
    triton_out = paged_attention(
        query, key, value, block_table,
        actual_seq_lengths1, actual_seq_lengths_kv1,
        num_heads, num_key_value_heads, scale,
        BLOCK_SIZE, block_m, block_n,
        sinks=None,
        atten_mask=atten_mask,
    )
    print(triton_out.shape, triton_out)
    base_cpu = triton_out.to(torch.float32).cpu()
    check_nan_inf(base_cpu, "triton_out")

    fias_out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale=scale,
        atten_mask=atten_mask,
        block_table=block_table,
        input_layout=input_layout,
        block_size=block_size,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_mode=sparse_mode,
    )
    torch.npu.synchronize()

    mse1 = (triton_out.float() - fias_out.float()).pow(2).mean().item()
    mse2 = (triton_out.float() - attn_output.float()).pow(2).mean().item()
    mse3 = (fias_out.float() - attn_output.float()).pow(2).mean().item()
    print(f"[triton-contiguous-prefill-vs-fias_v1] mse1: {mse1}, mse2: {mse2}, mse3: {mse3}")
    torch.testing.assert_close(triton_out, fias_out, atol=5e-3, rtol=5e-3)
    
    torch.testing.assert_close(attn_output, fias_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(triton_out, attn_output, atol=5e-3, rtol=5e-3)