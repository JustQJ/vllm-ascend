# Triton PagedAttention FULL Decode ACLGraph Design

## Objective

Add ACLGraph support for the existing Triton PagedAttention implementation on
Ascend A5, limited to ordinary single-token `DecodeOnly` batches.

The first phase explicitly excludes:

- MTP and speculative decoding;
- sliding-window attention;
- attention sinks;
- non-causal attention;
- contiguous-KV prefill;
- compressed, sparse, and C8 KV-cache paths.

Prefill, chunked prefill, and mixed batches must keep using the existing eager
Triton PagedAttention path. `FULL_AND_PIECEWISE` integration is a later phase.

## Existing Boundary

vLLM already registers `vllm::unified_attention_with_output` as an opaque Torch
custom op. Ascend reports `opaque_attention_op() == True`, so Dynamo records the
operator as one FX node and does not trace into
`AscendAttentionBackendImpl.forward()`.

Consequently, this feature does not add another Torch custom op. The existing
boundary is reused for both Dynamo opacity and piecewise attention splitting.

## Selected Architecture

Use a decode-specialized Triton out-kernel inside the existing ACLGraph task
group infrastructure. Capture and replay use device-resident metadata; no
Python length list is converted to an NPU tensor inside graph capture.

The high-level path is:

```text
torch.ops.vllm.unified_attention_with_output
  -> AscendAttentionBackendImpl.forward
  -> forward_impl
  -> forward_fused_infer_attention
  -> full_graph_triton_paged_attention
  -> graph_task_group
  -> paged_attention_decode_out
  -> Triton _paged_attn_fwd
```

The existing generic `paged_attention()` entry remains unchanged for eager
prefill and eager decode.

## Why Decode Does Not Need Dynamic Query-Block Metadata

For the supported mode, every sequence contributes exactly one query token.
For a full-graph gear of size `B`:

```text
query lengths       = [1, 1, ..., 1]
query blocks        = [1, 1, ..., 1]
q_block_seq         = [0, 1, ..., B - 1]
q_block_local       = [0, 0, ..., 0]
```

The decode-specialized kernel therefore derives the mapping directly:

```text
seq                 = program_id(0)
q_block_local       = 0
q_start             = seq
q_end               = seq + 1
q_len               = 1
```

This removes query-side `cumsum`, `repeat_interleave`, `q_block_seq`,
`q_block_local`, and `actual_seq_lengths_q` from the FULL decode launch.

The rule remains valid when the runtime batch is padded to a larger capture
gear: every real or padding request owns one query slot. Padding rows use a
zero KV length and a safe block-table row, produce zero output, and are trimmed
by the existing model-runner path.

## Device Metadata

Only two pieces of runtime attention metadata remain dynamic:

- per-sequence KV lengths;
- block-table contents.

The model runner already owns persistent NPU buffers for both:

- `NPUModelRunner.seq_lens`;
- the committed block-table tensor.

However, the current `AscendAttentionMetadataBuilder` stores the CPU
`seq_lens` tensor in `AscendMetadata.seq_lens`. To avoid changing existing FIA
semantics, add a separate field:

```python
seq_lens_device: torch.Tensor
```

Populate it from `common_attn_metadata.seq_lens[:num_reqs]`. Triton FULL decode
uses `seq_lens_device` and `block_tables`, both of which are prepared before
model execution. The Triton kernel accepts `int32` KV lengths and casts loaded
values to `tl.int64` internally.

No capture-time operation performs:

```python
torch.as_tensor(host_lengths, device="npu")
```

## Triton Out Interface

Add a dedicated inference-only entry such as:

```text
paged_attention_decode_out(
    query,
    key_cache,
    value_cache,
    block_table,
    kv_lens,
    output,
    static head and block parameters,
)
```

It must:

- accept a caller-provided `output` tensor;
- never allocate the final output internally;
- launch a fixed grid `(query.shape[0], num_q_heads)`;
- select the decode-specialized compile-time branch in `_paged_attn_fwd`;
- avoid dynamic-shape Torch operations before the Triton launch;
- support the `VLLM_ASCEND_PAGED_ATTN_USE_MXFP4_P` compile-time flag.

The generic eager wrapper continues to allocate and return its own output and
continues using the general query-block construction.

## Capture Flow

`forward_fused_infer_attention()` selects the FULL Triton path only when all
first-phase constraints hold. Otherwise, it falls back to the existing FIA
FULL-graph implementation.

During capture, `full_graph_triton_paged_attention()`:

1. Obtains the paged KV-cache views and block size.
2. Uses `attn_metadata.seq_lens_device` and `attn_metadata.block_tables`.
3. Creates an event and waits/resets it using the established FIA ordering.
4. Stores weak references to query, KV cache, output, mask if needed, and the
   exact layer name, plus static launch parameters.
5. Begins an NPU graph task group.
6. Calls `paged_attention_decode_out()` with the caller-provided output.
7. Ends the task group and stores its handle.
8. Returns the same output tensor.

There are no fixed padded metadata buffers and no device fill/arange kernels
outside the task group.

## Replay Update Flow

Triton graph tasks use storage separate from FIA `attn_params`, preventing tuple
layout collisions when a graph contains a fallback attention layer.

For each captured Triton task, `update_graph_params()`:

1. Uses the captured layer name to select current
   `forward_context.attn_metadata`.
2. Reads the current `seq_lens_device` and `block_tables` tensors.
3. Calls `graph_task_update_begin()`.
4. Reissues `paged_attention_decode_out()` using the saved query/cache/output
   tensors and current device metadata.
5. Calls `graph_task_update_end()` and records the external event.

No host length list is copied to the device. Task update rebinds current device
tensor addresses if a view changes; stable model-runner buffers also allow the
captured launch to observe refreshed contents.

The Triton update does not unconditionally return from `update_graph_params()`;
existing FIA/native task updates remain available for fallback layers.

## State Ownership

Add dedicated per-gear Triton task storage to `GraphParams`, rather than
overloading the existing FIA tuple list. Each task record contains:

- weak query, key-cache, value-cache, and output tensors;
- block size, head counts, scale, mask, and MXFP4-P flag;
- captured layer name;
- task-group handle and event.

Main, draft, and draft-prefill graph containers receive initialized storage,
although the first phase rejects speculative/draft execution.

No `triton_paged_bufs` field is required.

## Eager and Fallback Behavior

The current eager branch is restored to call the original generic
`paged_attention()` directly. It must not call a decode-only padded wrapper.

The FULL Triton predicate requires:

```text
VLLM_ASCEND_USE_PAGED_ATTENTION
and A5
and capturing
and DecodeOnly
and causal
and paged KV cache
and no speculative config
and no sliding window
and no sinks
and no C8/compressed/sparse path
```

Unsupported cases use the existing FIA/native implementation without adding a
partially compatible Triton task to the graph state.

## Files to Change

### `vllm_ascend/attention/attention_v1.py`

- Add `seq_lens_device` to `AscendMetadata` and populate it in the builder.
- Record the attention layer name for every Triton graph-capable layer.
- Add the first-phase support predicate.
- Add `full_graph_triton_paged_attention()`.
- Add the dedicated Triton replay-update loop using current layer metadata.
- Restore the eager environment-variable branch to generic
  `paged_attention()`.
- Remove nested `torch.ops.vllm.ascend_paged_attention_triton` calls.

### `vllm_ascend/ops/triton/paged_attn/paged_attention_npu.py`

- Add the decode-specialized constexpr path to `_paged_attn_fwd`.
- Add `paged_attention_decode_out()`.
- Accept device `int32` KV lengths in the decode path and cast in the kernel.
- Remove the incomplete graph padding helpers and padded autograd entry.
- Preserve the generic eager API and its existing tests.

### `vllm_ascend/compilation/acl_graph.py`

- Replace `triton_paged_bufs` with dedicated per-gear Triton task storage.
- Initialize it in main, draft, and draft-prefill graph containers.

### `vllm_ascend/ops/register_custom_ops.py`

- Remove the nested Triton PagedAttention custom-op import.

### `vllm_ascend/platform.py`

- Remove `vllm::ascend_paged_attention_triton` from splitting ops. The existing
  `vllm::unified_attention_with_output` boundary remains authoritative.

### Tests

- Add unit coverage for the support predicate, metadata propagation, output
  identity, and current-metadata replay selection.
- Add an NPU kernel test comparing `paged_attention_decode_out()` with the
  existing generic eager wrapper.
- Add an A5 E2E ACLGraph test comparing eager and `FULL_DECODE_ONLY` results
  over multiple decode steps and capture gears.

## Verification Matrix

The implementation is complete only after these checks pass on A5:

1. Existing eager prefill and decode Triton tests remain correct.
2. A prefill request succeeds under `FULL_DECODE_ONLY` before graph decode.
3. First FULL capture succeeds for at least two capture gears.
4. At least three consecutive replay steps use increasing KV lengths and match
   eager output within the existing tolerance.
5. A runtime batch smaller than its capture gear produces correct real-token
   output and harmless padding output.
6. The caller-provided output address is identical to the Triton kernel output
   address during capture and task update.
7. Unsupported sinks, sliding-window, speculative, and non-causal cases follow
   their existing fallback paths.
8. Both MXFP4-P disabled and enabled launches capture and replay successfully.
9. `git diff --check`, formatting, targeted unit tests, and the new A5 E2E test
   pass.

## Success Criteria

With `VLLM_ASCEND_USE_PAGED_ATTENTION=1` and
`cudagraph_mode=FULL_DECODE_ONLY` on A5:

- prefill remains eager and correct;
- ordinary uniform decode captures the Triton PagedAttention kernel in the FULL
  ACLGraph path;
- replay consumes current device KV lengths and block-table contents;
- outputs match eager Triton PagedAttention across multiple decode steps;
- no host-to-device length conversion occurs inside graph capture;
- unsupported attention configurations safely retain their existing backend.
