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
    DECODE_SPLIT_KV_NUM_PROGRAMS,
    build_split_kv_descriptors,
    select_decode_heads_per_program,
)

HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
BLOCK_M = 16
BLOCK_N = 64
SPLIT_KV_NUM_PROGRAMS = DECODE_SPLIT_KV_NUM_PROGRAMS
SOFTMAX_SCALE = HEAD_DIM**-0.5
SWA_INT_MAX = 2147483647
CSV_COLUMNS = [
    "num_q_heads",
    "batch_size",
    "kv_len",
    "kv_lens",
    "heads_per_program",
    "num_head_groups",
    "grid_size",
    "split_kv_num_programs",
    "split_kv_chunk_size",
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
    parser.add_argument("--q-heads", type=int, nargs="+", default=[16])
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
    parser.add_argument(
        "--heterogeneous-kv-lens",
        type=int,
        nargs="+",
        default=[65536, 8192, 1024, 0],
        help=(
            "Run one additional varied-length batch. The default includes a "
            "zero-length graph-padding sequence."
        ),
    )
    parser.add_argument(
        "--heterogeneous-only",
        action="store_true",
        help="Skip uniform-length cases and run only the heterogeneous batch.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--heads-per-program",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 8, 16],
        help=(
            "Override the decode Q-head grouping. Multiple values run an HPP "
            "sweep; omit this option to use the automatic policy."
        ),
    )
    parser.add_argument(
        "--split-kv-programs",
        type=int,
        nargs="+",
        choices=[1, SPLIT_KV_NUM_PROGRAMS],
        default=[1],
        help=(
            "Use 1 for the direct kernel or 32 for the fixed Split-KV "
            "program pool."
        ),
    )
    parser.add_argument("--use-mxfp4-p", action="store_true")
    return parser.parse_args()


def build_inputs(num_q_heads, kv_lens):
    batch_size = len(kv_lens)
    max_kv_len = max(kv_lens)
    blocks_per_sequence = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch_size * blocks_per_sequence
    query = torch.randn(
        batch_size,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    key_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    value_cache = torch.randn_like(key_cache)

    block_table = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size,
        blocks_per_sequence,
    )
    block_table[::2] = block_table[::2].flip(1)
    block_table = block_table.to("npu").contiguous()
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int64, device="npu")
    cumulative_q_lens = torch.arange(
        1,
        batch_size + 1,
        dtype=torch.int64,
        device="npu",
    )
    causal_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.int8, device="npu"),
        diagonal=1,
    ).contiguous()
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "kv_lens": kv_lens_tensor,
        "kv_lens_cpu": kv_lens,
        "cumulative_q_lens": cumulative_q_lens,
        "causal_mask": causal_mask,
        "output": torch.empty_like(query),
    }


def make_backend_functions(
    inputs,
    num_q_heads,
    use_mxfp4_p,
    heads_per_program_override,
    split_kv_num_programs,
):
    query = inputs["query"]
    key_cache = inputs["key_cache"]
    value_cache = inputs["value_cache"]
    block_table = inputs["block_table"]
    kv_lens = inputs["kv_lens"]
    kv_lens_cpu = inputs["kv_lens_cpu"]
    cumulative_q_lens = inputs["cumulative_q_lens"]
    causal_mask = inputs["causal_mask"]
    output = inputs["output"]
    batch_size = query.shape[0]
    split_kv_workspace = None
    split_kv_descriptors = None
    if split_kv_num_programs > 1:
        partial_output = torch.empty(
            SPLIT_KV_NUM_PROGRAMS,
            num_q_heads,
            HEAD_DIM,
            dtype=torch.float32,
            device=query.device,
        )
        partial_lse = torch.empty(
            SPLIT_KV_NUM_PROGRAMS,
            num_q_heads,
            dtype=torch.float32,
            device=query.device,
        )
        split_kv_workspace = (partial_output, partial_lse)
        work_desc, seq_desc, _ = build_split_kv_descriptors(
            kv_lens_cpu,
            block_size=BLOCK_SIZE,
        )
        split_kv_descriptors = (
            torch.tensor(work_desc, dtype=torch.int32, device=query.device),
            torch.tensor(seq_desc, dtype=torch.int32, device=query.device),
        )

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
            num_q_heads=num_q_heads,
            num_kv_heads=NUM_KV_HEADS,
            use_mxfp4_p=use_mxfp4_p,
            heads_per_program_override=heads_per_program_override,
            split_kv_num_programs=split_kv_num_programs,
            split_kv_workspace=split_kv_workspace,
            split_kv_descriptors=split_kv_descriptors,
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
            actual_seq_lengths_kv=kv_lens_cpu,
            num_key_value_heads=NUM_KV_HEADS,
            num_heads=num_q_heads,
            scale=SOFTMAX_SCALE,
            sparse_mode=3,
            pre_tokens=SWA_INT_MAX,
            next_tokens=SWA_INT_MAX,
        )
        return result

    return {"specialized": specialized, "generic": generic, "fia": fia}


def check_correctness(backends, use_mxfp4_p, kv_lens):
    specialized = backends["specialized"]()
    generic = backends["generic"]()
    torch.npu.synchronize()
    torch.testing.assert_close(specialized, generic, atol=5e-3, rtol=5e-3)
    padding_indices = [index for index, kv_len in enumerate(kv_lens) if kv_len == 0]
    if padding_indices:
        padding_output = specialized[padding_indices]
        torch.testing.assert_close(
            padding_output,
            torch.zeros_like(padding_output),
            atol=0,
            rtol=0,
        )
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
    kv_lens,
    use_mxfp4_p,
    warmup,
    samples,
    repeats,
    heads_per_program_override,
    split_kv_num_programs,
):
    batch_size = len(kv_lens)
    inputs = build_inputs(num_q_heads, kv_lens)
    heads_per_program = heads_per_program_override
    if heads_per_program is None:
        heads_per_program = select_decode_heads_per_program(batch_size, num_q_heads, 32)
    split_kv_chunk_size = 0
    if split_kv_num_programs > 1:
        _, _, split_kv_chunk_size = build_split_kv_descriptors(
            kv_lens,
            block_size=BLOCK_SIZE,
        )
        heads_per_program = num_q_heads
    backends = make_backend_functions(
        inputs,
        num_q_heads,
        use_mxfp4_p,
        heads_per_program_override,
        split_kv_num_programs,
    )
    check_correctness(backends, use_mxfp4_p, kv_lens)
    latencies = {
        name: measure_latency_us(function, warmup, samples, repeats)
        for name, function in backends.items()
    }

    num_head_groups = num_q_heads // heads_per_program
    generic_median = latencies["generic"]["median"]
    fia_median = latencies["fia"]["median"]
    rows = []
    for backend, timing in latencies.items():
        rows.append(
            {
                "num_q_heads": num_q_heads,
                "batch_size": batch_size,
                "kv_len": max(kv_lens),
                "kv_lens": list(kv_lens),
                "heads_per_program": heads_per_program,
                "num_head_groups": num_head_groups,
                "grid_size": (
                    SPLIT_KV_NUM_PROGRAMS
                    if split_kv_num_programs > 1
                    else batch_size * num_head_groups
                ),
                "split_kv_num_programs": split_kv_num_programs,
                "split_kv_chunk_size": split_kv_chunk_size,
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


def run_and_print_case(
    *,
    num_q_heads,
    kv_lens,
    use_mxfp4_p,
    warmup,
    samples,
    repeats,
    heads_per_program_override,
    split_kv_num_programs,
):
    rows = benchmark_case(
        num_q_heads=num_q_heads,
        kv_lens=kv_lens,
        use_mxfp4_p=use_mxfp4_p,
        warmup=warmup,
        samples=samples,
        repeats=repeats,
        heads_per_program_override=heads_per_program_override,
        split_kv_num_programs=split_kv_num_programs,
    )
    print(
        "Benchmark result for "
        f"num_q_heads={num_q_heads}, "
        f"batch_size={len(kv_lens)}, "
        f"kv_lens={list(kv_lens)}, "
        f"heads_per_program={rows[0]['heads_per_program']}, "
        f"split_kv_num_programs={split_kv_num_programs}, "
        f"split_kv_chunk_size={rows[0]['split_kv_chunk_size']}:"
    )
    for row in rows:
        print(row)


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
    heterogeneous_kv_lens = args.heterogeneous_kv_lens
    if len(heterogeneous_kv_lens) not in (1, 2, 4):
        raise ValueError("--heterogeneous-kv-lens must contain 1, 2, or 4 values")
    if any(length < 0 for length in heterogeneous_kv_lens):
        raise ValueError("--heterogeneous-kv-lens must be non-negative")
    if max(heterogeneous_kv_lens) == 0:
        raise ValueError("--heterogeneous-kv-lens requires a non-padding sequence")
    if args.heads_per_program is not None:
        invalid_pairs = [
            (num_q_heads, heads_per_program)
            for num_q_heads in args.q_heads
            for heads_per_program in args.heads_per_program
            if num_q_heads % heads_per_program != 0
        ]
        if invalid_pairs:
            raise ValueError(
                "--heads-per-program must divide --q-heads; invalid pairs: "
                f"{invalid_pairs}"
            )

    torch.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)
    heads_per_program_values = args.heads_per_program or [None]
    for num_q_heads in args.q_heads:
        uniform_batch_sizes = [] if args.heterogeneous_only else args.batch_sizes
        for batch_size in uniform_batch_sizes:
            for kv_len in args.kv_lens:
                if kv_len >= 32768 and batch_size >= 8:
                    print(
                        "Skipping case: "
                        f"num_q_heads={num_q_heads}, "
                        f"batch_size={batch_size}, kv_len={kv_len} "
                        "(too large for memory)"
                    )
                    continue

                for heads_per_program in heads_per_program_values:
                    for split_kv_num_programs in args.split_kv_programs:
                        if split_kv_num_programs > 1 and batch_size not in (1, 2, 4):
                            print(
                                "Skipping fixed Split-KV case: "
                                f"batch_size={batch_size} is not a graph gear"
                            )
                            continue
                        run_and_print_case(
                            num_q_heads=num_q_heads,
                            kv_lens=[kv_len] * batch_size,
                            use_mxfp4_p=args.use_mxfp4_p,
                            warmup=args.warmup,
                            samples=args.samples,
                            repeats=args.repeats,
                            heads_per_program_override=heads_per_program,
                            split_kv_num_programs=split_kv_num_programs,
                        )

        for heads_per_program in heads_per_program_values:
            for split_kv_num_programs in args.split_kv_programs:
                run_and_print_case(
                    num_q_heads=num_q_heads,
                    kv_lens=heterogeneous_kv_lens,
                    use_mxfp4_p=args.use_mxfp4_p,
                    warmup=args.warmup,
                    samples=args.samples,
                    repeats=args.repeats,
                    heads_per_program_override=heads_per_program,
                    split_kv_num_programs=split_kv_num_programs,
                )


if __name__ == "__main__":
    main()
