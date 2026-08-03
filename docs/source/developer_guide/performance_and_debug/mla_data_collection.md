# MLA Q and Latent-KV Data Collection

This debug-only collector saves bounded samples of the Q and latent-KV tensors used by MLA attention. It is intended
for offline KV-cache quantization analysis. Collection is disabled by default and introduces device synchronization and
file I/O when enabled.

## Collected tensors

Each dump contains the current tokens for one layer and one phase (`prefill` or `decode`):

- `kv_latent_pre_norm`: the 512-dimensional latent before KV RMSNorm;
- `kv_latent`: the latent after KV RMSNorm and before it is written to the paged cache;
- `k_rope_pre` and `k_rope_post`: the decoupled RoPE key before and after RoPE;
- `q_latent`: Q after absorbing `W_UK`, in the same latent space as `kv_latent`;
- `q_rope_pre` and `q_rope_post`: the RoPE query before and after RoPE.
- `positions`: the absolute token positions aligned with the saved tensor samples.

## Usage

Select a small set of shallow, middle, and deep layers first to bound disk usage. For example:

```bash
export VLLM_ASCEND_MLA_DATA_DUMP_DIR=/workspace/mla-data
export VLLM_ASCEND_MLA_DATA_DUMP_LAYERS=0,30,60
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS=128
export VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS=1

vllm serve /models/DeepSeek-V3.1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager
```

The collector skips ACL Graph capture. Use eager mode for deterministic calibration runs. Following OSCAR's collection
pattern, every tensor-parallel rank participates in the Q tensor all-gather, but only tensor-parallel rank 0 copies the
result to CPU and writes a file. Q tensors are gathered across the head dimension. MLA latent KV, RoPE K, and positions
are already replicated across tensor-parallel ranks, so rank 0 saves its complete local copy without duplicating it.
Dump files contain a `torch.save` payload with metadata and a `tensors` dictionary. The leading token dimension is
truncated to `VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS`.

With data or pipeline parallelism, this means one writer for each tensor-parallel group rather than one writer for the
entire distributed job, so independent request or pipeline-stage data is not discarded.

The dump is derived from request data. Store it in a protected directory and remove it after analysis.

## Limitations

- The collection path is intended for BF16/FP16 MLA calibration. Fused MLAPO/quantized decode preprocess may bypass the
  Python tensor points; disable MLAPO and use an unquantized checkpoint when collecting the baseline distribution.
- Context-parallel attention uses a separate implementation and is not collected by this initial path.
- This collector saves samples rather than the full paged KV cache. Run representative prompts to cover text, code,
  mathematics, and long-context retrieval distributions.
