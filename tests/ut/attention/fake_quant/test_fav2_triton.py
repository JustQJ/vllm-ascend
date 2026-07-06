"""Triton attention (with mxfp4c7) vs torch golden (with mxfp4c7) precision comparison.

Compares:
  - Triton _attn_fwd kernel (to_mxfp4c7 applied to softmax weights p)
  - Torch golden: block-wise online softmax + _mxfp4_quant_tf (c7 variant) on p

Both sides apply the same mxfp4c7 quantization, so the diff should be small
(mainly from tl.dot vs torch.matmul accumulation precision).
"""
import torch
import torch_npu
import triton
import triton.language as tl
from test_npu_fused_infer_attention_triton import attention

# ---------------------------------------------------------------------------
# MXFP4 constants (from user-provided algorithm)
# ---------------------------------------------------------------------------
_MXFP4_EBITS = 2
_MXFP4_MBITS = 3
_MXFP4_EMAX = 2
_MXFP4_MAX_NORM = 6.0
_MXFP4_BLOCK_SIZE = 32
_MXFP4_MIN_EXP = 0.0
_MXFP4_SCALE_FACTOR = 2.0
_MXFP4_INV_SCALE_FACTOR = 0.5
_MXFP4_EPSILON = 1.17e-38
_E8M0_SCALE_EMAX = 127


def _mxfp4_quant_tf(x, qdim, stochastic_rounding=False):
    """MXFP4-C7 quantization → dequantized (recovered) tensor.

    Matches triton to_mxfp4c7(p_cx=7): shared_exp = ceil(log2(max_val / 7)).
    Fixed from user-provided code (variable name/ndim bugs corrected).
    """
    ndim = x.ndim
    orig_shape = x.shape
    normalized_qdim = qdim if qdim >= 0 else ndim + qdim
    reduction_dim = normalized_qdim + 1
    x = x.unflatten(qdim, (-1, _MXFP4_BLOCK_SIZE))

    max_val = torch.amax(x.abs(), reduction_dim, keepdim=True)
    inv_constant = 1 / 7  # c7: divide max by 7 before computing shared exp
    shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))
    shared_exp = shared_exp.clamp(-127, 127)

    x = x * torch.exp2(-shared_exp)

    private_exp = torch.floor(torch.log2(x.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)
    x = x * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR
    if stochastic_rounding:
        x.add_(torch.rand_like(x)).floor_()
    else:
        x_sign = torch.sign(x)
        x = x_sign * torch.floor_(x.abs() + 0.5)

    x = (x * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)
    recovered = x * torch.exp2(shared_exp)
    return recovered.reshape(orig_shape)


def golden_attn_mxfp4c7(q, k, v, sm_scale, BLOCK_M, BLOCK_N):
    """Torch golden attention with mxfp4c7, replicating triton block-wise online softmax.

    q/k/v: (Z, H, N_CTX, HEAD_DIM) BNSD, non-causal.
    Iterates BLOCK_M rows × BLOCK_N cols, same as _attn_fwd_inner STAGE=3.
    """
    Z, H, N_CTX, HEAD_DIM = q.shape
    dtype = q.dtype
    device = q.device
    out = torch.zeros((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)

    for z in range(Z):
        for h in range(H):
            for mb in range(N_CTX // BLOCK_M):
                q_blk = q[z, h, mb * BLOCK_M:(mb + 1) * BLOCK_M].to(torch.float32)  # (BM, D)

                m_i = torch.full((BLOCK_M,), float('-inf'), device=device)
                l_i = torch.zeros((BLOCK_M,), device=device)
                acc = torch.zeros((BLOCK_M, HEAD_DIM), device=device)

                for nb in range(0, N_CTX, BLOCK_N):
                    k_blk = k[z, h, nb:nb + BLOCK_N].to(torch.float32)  # (BN, D)
                    v_blk = v[z, h, nb:nb + BLOCK_N].to(torch.float32)

                    qk = torch.matmul(q_blk, k_blk.T) * sm_scale  # (BM, BN)

                    m_ij = torch.maximum(m_i, qk.max(dim=1).values)
                    qk = qk - m_ij[:, None]

                    p = torch.exp(qk)                      # float32, unnormalized
                    p = _mxfp4_quant_tf(p, qdim=-1)        # mxfp4c7 quantize→dequantize
                    p = p.to(dtype).to(torch.float32)      # cast to dtype (match triton .to(tl.float16))

                    pv = torch.matmul(p, v_blk)            # (BM, D)
                    l_ij = p.sum(dim=1)

                    alpha = torch.exp(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None] + pv
                    m_i = m_ij

                out[z, h, mb * BLOCK_M:(mb + 1) * BLOCK_M] = (acc / l_i[:, None]).to(dtype)

    return out


def golden_attn_no_quant(q, k, v, sm_scale):
    """Standard attention without mxfp4c7 (reference for quantization error)."""
    qk = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)) * sm_scale
    attn = torch.softmax(qk, dim=-1)
    out = torch.matmul(attn, v.to(torch.float32))
    return out.to(q.dtype)


def test_mxfp4c7_precision(Z=1, H=1, N_CTX=128, HEAD_DIM=128, dtype=torch.float16,
                           BM=32, BN=128, sm_scale=0.5):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="npu").normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="npu").normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="npu").normal_(mean=0.0, std=0.5)

    # 1. Triton attention (with mxfp4c7)
    tri_out = attention(q, k, v, False, sm_scale, BM, BN)

    # 2. Torch golden (with mxfp4c7) — validates triton mxfp4c7 implementation
    golden_q_out = golden_attn_mxfp4c7(q, k, v, sm_scale, BM, BN)

    # 3. Standard attention (no quantization) — reference for quantization error magnitude
    golden_noq_out = golden_attn_no_quant(q, k, v, sm_scale)

    tri_f = tri_out.to(torch.float32)
    gold_q_f = golden_q_out.to(torch.float32)
    gold_noq_f = golden_noq_out.to(torch.float32)

    diff_vs_quant = (tri_f - gold_q_f).abs()
    diff_vs_noq = (tri_f - gold_noq_f).abs()

    print(f"\n{'='*60}")
    print(f"[MXFP4C7 Precision Comparison]")
    print(f"  config : Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, dtype={dtype}")
    print(f"  blocks : BLOCK_M={BM}, BLOCK_N={BN}, sm_scale={sm_scale}")
    print(f"{'-'*60}")
    print(f"  triton out   (first 8): {tri_f.flatten()[:8].tolist()}")
    print(f"  golden+quant (first 8): {gold_q_f.flatten()[:8].tolist()}")
    print(f"  golden no-q  (first 8): {gold_noq_f.flatten()[:8].tolist()}")
    print(f"{'-'*60}")
    print(f"  [triton vs golden+quant]  max={diff_vs_quant.max().item():.6f}  mean={diff_vs_quant.mean().item():.6f}")
    print(f"  [triton vs golden no-q ]  max={diff_vs_noq.max().item():.6f}  mean={diff_vs_noq.mean().item():.6f}")

    # triton vs golden+quant: should be small (validates mxfp4c7 correctness)
    assert diff_vs_quant.max().item() < 0.1, \
        f"triton vs golden+quant max_diff={diff_vs_quant.max().item()} too large"
    print(f"  [PASS] triton vs golden+quant max_diff < 0.1")


if __name__ == "__main__":
    test_mxfp4c7_precision()
