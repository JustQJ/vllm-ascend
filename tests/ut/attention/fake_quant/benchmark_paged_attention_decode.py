"""Benchmark specialized Triton decode against generic Triton and FIA.

Run this script on an Ascend NPU with the vLLM Ascend runtime installed.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.triton.paged_attn import paged_attention, paged_attention_decode_out
from vllm_ascend.ops.triton.paged_attn.decode_utils import (
    select_decode_heads_per_program,
)

HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
BLOCK_M = 16
BLOCK_N = 64
SOFTMAX_SCALE = HEAD_DIM**-0.5
SWA_INT_MAX = 2147483647
CSV_COLUMNS = [
    "num_q_heads",
    "batch_size",
    "kv_len",
    "heads_per_program",
    "num_head_groups",
    "grid_size",
    "use_mxfp4_p",
    "backend",
    "latency_us_min",
    "latency_us_median",
    "latency_us_max",
    "speedup_vs_generic",
    "speedup_vs_fia",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-heads", type=int, nargs="+", default=[8, 16])
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--kv-lens",
        type=int,
        nargs="+",
        default=[128, 1024, 4096, 8192, 16384, 32768, 40960],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--use-mxfp4-p", action="store_true")
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def build_inputs(batch_size, num_q_heads, kv_len):
    blocks_per_sequence = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch_size * blocks_per_sequence
    query = torch.randn(
        batch_size,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    key_cache = torch.randn(
        32768*16 // BLOCK_SIZE,
        BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    value_cache = torch.randn_like(key_cache)

    block_table = torch.arange(num_blocks, dtype=torch.int32).view(batch_size, blocks_per_sequence)
    block_table[::2] = block_table[::2].flip(1)
    block_table = block_table.to("npu").contiguous()
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int64, device="npu")
    cumulative_q_lens = torch.arange(1, batch_size + 1, dtype=torch.int64, device="npu")
    causal_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.int8, device="npu"),
        diagonal=1,
    ).contiguous()
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "kv_lens": kv_lens,
        "cumulative_q_lens": cumulative_q_lens,
        "causal_mask": causal_mask,
        "output": torch.empty_like(query),
    }


def make_backend_functions(inputs, num_q_heads, kv_len, use_mxfp4_p):
    query = inputs["query"]
    key_cache = inputs["key_cache"]
    value_cache = inputs["value_cache"]
    block_table = inputs["block_table"]
    kv_lens = inputs["kv_lens"]
    cumulative_q_lens = inputs["cumulative_q_lens"]
    causal_mask = inputs["causal_mask"]
    output = inputs["output"]
    batch_size = query.shape[0]

    def specialized():
        return paged_attention_decode_out(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            actual_seq_qlen=cumulative_q_lens,
            actual_seq_kvlen=kv_lens,
            output=output,
            softmax_scale=SOFTMAX_SCALE,
            block_size=BLOCK_SIZE,
            use_mxfp4_p=use_mxfp4_p,
        )

    def generic():
        return paged_attention(
            query,
            key_cache,
            value_cache,
            block_table,
            cumulative_q_lens,
            kv_lens,
            num_q_heads,
            NUM_KV_HEADS,
            SOFTMAX_SCALE,
            BLOCK_SIZE,
            BLOCK_M,
            BLOCK_N,
            sinks=None,
            atten_mask=causal_mask,
            use_mxfp4_p=use_mxfp4_p,
        )

    def fia():
        result, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=causal_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=BLOCK_SIZE,
            actual_seq_lengths=list(range(1, batch_size + 1)),
            actual_seq_lengths_kv=[kv_len] * batch_size,
            num_key_value_heads=NUM_KV_HEADS,
            num_heads=num_q_heads,
            scale=SOFTMAX_SCALE,
            sparse_mode=3,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )
        return result

    return {"specialized": specialized, "generic": generic, "fia": fia}


def check_correctness(backends, use_mxfp4_p):
    specialized = backends["specialized"]()
    generic = backends["generic"]()
    torch.npu.synchronize()
    torch.testing.assert_close(specialized, generic, atol=5e-3, rtol=5e-3)
    if not use_mxfp4_p:
        fia = backends["fia"]()
        torch.npu.synchronize()
        torch.testing.assert_close(specialized, fia, atol=2e-2, rtol=2e-2)


def measure_latency_us(function, warmup, samples, repeats):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function()
        end.record()
        torch.npu.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / repeats)
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "max": max(timings),
    }


def benchmark_case(
    num_q_heads,
    batch_size,
    kv_len,
    use_mxfp4_p,
    warmup,
    samples,
    repeats,
):
    inputs = build_inputs(batch_size, num_q_heads, kv_len)
    backends = make_backend_functions(inputs, num_q_heads, kv_len, use_mxfp4_p)
    check_correctness(backends, use_mxfp4_p)
    latencies = {name: measure_latency_us(function, warmup, samples, repeats) for name, function in backends.items()}

    heads_per_program = select_decode_heads_per_program(batch_size, num_q_heads, 32)
    num_head_groups = num_q_heads // heads_per_program
    generic_median = latencies["generic"]["median"]
    fia_median = latencies["fia"]["median"]
    rows = []
    for backend, timing in latencies.items():
        rows.append(
            {
                "num_q_heads": num_q_heads,
                "batch_size": batch_size,
                "kv_len": kv_len,
                "heads_per_program": heads_per_program,
                "num_head_groups": num_head_groups,
                "grid_size": batch_size * num_head_groups,
                "use_mxfp4_p": use_mxfp4_p,
                "backend": backend,
                "latency_us_min": f"{timing['min']:.3f}",
                "latency_us_median": f"{timing['median']:.3f}",
                "latency_us_max": f"{timing['max']:.3f}",
                "speedup_vs_generic": f"{generic_median / timing['median']:.3f}",
                "speedup_vs_fia": f"{fia_median / timing['median']:.3f}",
            }
        )
    return rows


def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required")
    if args.warmup < 0 or args.samples <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; samples/repeats must be positive")
    if any(heads not in (8, 16) for heads in args.q_heads):
        raise ValueError("--q-heads supports only 8 and 16")
    if any(size <= 0 for size in args.batch_sizes):
        raise ValueError("--batch-sizes must be positive")
    if any(length <= 0 for length in args.kv_lens):
        raise ValueError("--kv-lens must be positive")

    torch.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    for num_q_heads in args.q_heads:
        for batch_size in args.batch_sizes:
            for kv_len in args.kv_lens:
                if kv_len >= 32768 and batch_size >= 8:
                    print(f"Skipping case: num_q_heads={num_q_heads}, batch_size={batch_size}, kv_len={kv_len} (too large for memory)")
                    continue
                
                res = benchmark_case(
                        num_q_heads=num_q_heads,
                        batch_size=batch_size,
                        kv_len=kv_len,
                        use_mxfp4_p=args.use_mxfp4_p,
                        warmup=args.warmup,
                        samples=args.samples,
                        repeats=args.repeats,
                    )
                print(f"Benchmark result for num_q_heads={num_q_heads}, batch_size={batch_size}, kv_len={kv_len}:")
                for row in res:
                    print(row)

    


if __name__ == "__main__":
    main()
