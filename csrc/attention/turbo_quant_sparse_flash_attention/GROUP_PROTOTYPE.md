# Scheme A query grouping prototype

This is an opt-in implementation of exact union plus per-query membership.
It has not been compiled with CANN or validated on NPU hardware. The existing
framework entry point continues to run legacy SFA.

## Entry points and scope

- `torch.ops._C_ascend.turboquant_sparse_flash_attention`: existing default.
- `torch.ops._C_ascend.turboquant_sparse_flash_attention_grouped`: experimental
  entry with the same arguments, output shapes and Meta implementation.

The experimental entry uses two internal operators in the same stream:

1. `TurboQuantGroupPrepare`: device length reads, request-local windows, bounded
   integer hashing, union IDs, owner bits, and group descriptors.
2. `TurboQuantSparseFlashAttention`: grouped MM1/softmax/MM2 or singleton legacy
   execution selected from those descriptors.

There are no Python framework, scheduler, indexer or attention metadata changes.
The internal SFA ACLNN signature adds three optional tensors; rebuild both the
custom CANN operators and the PyTorch extension together. An old SFA binary is
not compatible with the new launcher, including its legacy entry point.

The initial grouped path requires BF16 TND queries, PA_BSND TQ4 KV, 16 query
heads, one KV head, token-level sparse indices and `1 <= K <= 2048`. Static
unsupported shapes or excessive metadata capacity use the original kernel.
Both sparse modes 0 and 3 retain their existing visibility semantics.

## Device planning

Each request is divided into windows of at most four queries. Tail lengths are
handled as `3 -> 2+1`, `2 -> 2`, and `1 -> 1`; windows never cross requests.
Quad and pair union caps are 3072 and 2560. These are prototype capacity gates,
not measured performance thresholds. Overflow is checked before a write and
causes `4 -> 2 -> 1` fallback, never union truncation. Duplicate valid IDs within
one query cause that query to use legacy multiplicity semantics.

IDs stay int32 throughout hashing, including IDs larger than the exact float32
integer range. Negative padding and positions outside the query's visibility
are excluded from ownership. The existing public input contract still applies:
lengths must be valid, block-table IDs must be in bounds, and padding must be
trailing. Invalid cumulative lengths are not an additional supported input mode.

The planner uses bulk GM-to-UB copies followed by scalar integer hash probes in
UB. This is a functional prototype; vectorized planning remains future work.
The exact `PlanWindow`/`BuildUnion` templates also run in the CPU tests.

Descriptors are `[T_capacity, 8]` int32. Union IDs and owners each reserve
`[T_capacity, K]`, with int32 and uint8 dtype respectively. A group beginning at
token `t` owns the range beginning at `t*K`, of capacity `groupSize*K`.
Group leaders are GROUP/LEGACY/ZERO; other slots are SKIP. Descriptors are fully
rewritten each invocation, including capacity-only trailing rows.

For `T=8192, K=2048`, metadata occupies 80.25 MiB. The launcher limits this to
128 MiB; normal SFA workspace and per-core planner UB are additional. Capture
pool reuse and peak memory across multiple layers still need on-device checks.

## Attention changes

The original `gSize` remains the query head count. Actual M is 16, 32 or 64.
One AIC and its two AIVs own each group; no group-level S2 split is introduced.
Four-query address ranges are distributed over cores to avoid leaving most
cores idle when group leaders lie at token indices 0, 4, 8, etc.

Union columns retain their logical order during KV copy. The legacy two-slot
physical-address reordering is bypassed for groups, so IDs, owners, RoPE and
TQ4 scale remain aligned. Each union column is loaded/dequantized once per group.

Each vector softmax block stays within one query's 16 heads. Owner bits encode
that query's effective original selection, including its causal filtering.
All-mask rows produce zero probabilities and preserve the previous online
max/sum/output with rescale factor 1. Empty final rows have output zero and
softmax sum zero. Normalization uses a safe denominator without changing LSE.

The existing tile pipeline and four-slot KV/scale ring are reused. This version
drains the final two stages at every group boundary; overlapping consecutive
groups and tuning load balance are deferred until correctness is established.

## Graph contract

Launch order, tensor capacities, addresses, tiling and core count depend only
on static shapes and attributes. Lengths, union lengths, ownership and 4/2/1
fallback are device data, recomputed on every replay. No `.item()`, CPU length
transfer, dynamic host allocation from union length, or host fallback occurs.

The NPU tests capture the experimental entry once, then replay it with
`[4,4]`, `[3,5]`, `[1,7]`, `[5,3]`, `[3,0]`, `[0,0]` and changing
overlap/duplicate patterns, including capacity-only padding rows.
This is a test to run, not evidence that graph replay already passes.
The framework's current SFA capture metadata supports DecodeOnly/SpecDecoding;
this patch does not add full-graph prefill support to the framework.

## Validation

The CPU planner test needs only Python's standard library and clang++ or g++:

```bash
python3 tests/ut/attention/test_turboquant_group_plan.py
```

It builds with AddressSanitizer and UndefinedBehaviorSanitizer, verifies exact
membership and independent scalar attention, and covers tails through 4097,
empty rows, hash collisions, large IDs, duplicate multiplicity, capacity
boundaries, metadata reuse and 2000 randomized cases.

After rebuilding on A2/A3, run:

```bash
pytest -sv tests/ut/attention/a2/test_turboquant_custom_ops.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_turboquant_custom_ops.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_turboquant_grouped_sfa.py
```

Compare complete two-op time against the byte-LUT legacy baseline on identical
inputs. Shared unique TopK measures the best-case grouping scenario; random
TopK may trigger duplicate fallback:

```bash
python tests/e2e/nightly/single_node/ops/singlecard_ops/turboquant_custom_ops_perf.py --ops sfa --sfa-implementation legacy --sfa-topk-pattern shared
python tests/e2e/nightly/single_node/ops/singlecard_ops/turboquant_custom_ops_perf.py --ops sfa --sfa-implementation grouped --sfa-topk-pattern shared
```

Remaining acceptance work: CANN compilation and API checks, NPU memory/event
sanitizers, output and LSE precision, graph replay, real overlap traces, planner
cost, workspace reuse, decode regressions and end-to-end model validation.
Only then consider switching the existing framework entry to the grouped
launcher and adding measured performance gates.
