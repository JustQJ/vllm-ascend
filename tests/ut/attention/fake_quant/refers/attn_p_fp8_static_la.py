"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl


@triton.jit
def to_fp8_static(tensor):
    tensor = tensor.to(tl.float32)
    S = tl.where(tensor < 0.0, -1.0, 1.0)
    # max_val = tl.max(tl.abs(tensor))
    # scale = max_val / 448.0  # approx max pos value of E4M3 is ~240
    scale = 1.0 / 448.0 + 1e-10 # kdn
    # Quantize to FP8 (fake)
    x_scaled = tensor / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)  # E4M3 range simulation
    tmp = tl.abs(x_clipped)
    E = tl.floor(tl.log2(tmp + 2.0 ** (-100)))
    res_abs =  tl.floor(tmp * tl.exp2(-E + 3) + 0.5) * tl.exp2(E - 3)
    # res_abs = S*res_abs
    return (S*res_abs*scale).to(tl.float16)



@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, qm, kv_len,
                    K_ptrs, K_fp_ptrs, V_ptrs, Mask_ptrs,
                    stride_kn, stride_vn, stride_kfpn, stride_maskn,
                    start_m, sm_scale: tl.constexpr,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask = k_mask)
        k_fp = tl.load(K_fp_ptrs, mask = k_mask)

        s_delta = tl.dot(qm, k_fp).to(tl.float32)
        qk = tl.dot(q, k) + s_delta
        qk_scale = sm_scale * 1.4426950408889634 # head_dim^-0.5 * 1/ln(2)
        qk *= qk_scale
        qk = qk.to(tl.float16)  ########## la


        # Load and Apply attention mask
        attn_mask = tl.load(Mask_ptrs, mask = k_mask)
        # qk = qk + tl.where(attn_mask, 0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1).to(tl.float32)


        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        # p = p.to(tl.float16)
        p = to_fp8_static(p).to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_fp_ptrs += BLOCK_N * stride_kfpn # 更新K_fp指针
        V_ptrs += BLOCK_N * stride_vn
        Mask_ptrs += BLOCK_N * stride_maskn # 更新Mask指针
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Qm, K_fp, Mask, Out, Lse,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              stride_qmz, stride_qmh, stride_qmn,
              stride_kfpz, stride_kfph, stride_kfpn,
              stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              sm_scale: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)  # 当前batch的起始地址
    off_h = tl.program_id(1).to(tl.int64)  # 当前head的起始地址

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # 行索引
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM) # 特征索引

    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Qm_ptrs = Qm + (off_z * stride_qmz + off_h * stride_qmh) + offs_m[:, None] * stride_qmn + offs_k[None, :]

    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_fp_ptrs = K_fp + (off_z * stride_kfpz + (off_h // num_kv_groups) * stride_kfph) + offs_n[None, :] * stride_kfpn + offs_k[:, None]

    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]

    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    # Mask pointer
    Mask_ptrs = Mask + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    qm = tl.load(Qm_ptrs, mask = offs_m[:, None] < qo_len) # 读取qm
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, qm, kv_len,
                                    K_ptrs, K_fp_ptrs, V_ptrs, Mask_ptrs,
                                    stride_kn, stride_vn, stride_kfpn, stride_maskn,
                                    start_m, sm_scale,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

def forward(q, k, v, qm, k_fp, attn_mask=None, sm_scale=1.0, tensor_layout="HND", output_dtype=torch.float16, return_lse=False):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
        stride_bz_qm, stride_h_qm, stride_seq_qm = qm.stride(0), qm.stride(1), qm.stride(2)
        stride_bz_k_fp, stride_h_k_fp, stride_seq_k_fp = k_fp.stride(0), k_fp.stride(1), k_fp.stride(2)

        if attn_mask is not None:
            # Only support mask shape with [b, 1, 1, seq_len]
            attn_mask = attn_mask.expand(b, 1, 1, kv_len).contiguous()
            stride_maskn = attn_mask.stride(-1)
        else:
            # Create default mask with all zeros (no masking)
            attn_mask = torch.zeros((b, 1, 1, kv_len), dtype=torch.float16, device=q.device)
            stride_maskn = attn_mask.stride(-1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, qm, k_fp, attn_mask, o, lse,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_qm, stride_h_qm, stride_seq_qm,
        stride_bz_k_fp, stride_h_k_fp, stride_seq_k_fp,
        stride_maskn,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 2)
        # 若num_stages设置不当，会报如下错误：
        # triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 188416, Hardware limit: 166912. Reducing block sizes or `num_stages` may help.

    return o, lse
