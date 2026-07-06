
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
# from torch_npu.testing.testcase import TestCase, run_tests


def golden_op_exec( query, key, value):
    # query/key/value: (T, N, D) TND 布局
    # 转成 (1, N, T, D) 走标准注意力计算作为参考值
    q = query.unsqueeze(0)
    k = key.unsqueeze(0)
    v = value.unsqueeze(0)
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / 0.0078125
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, v)
    return attn_output.squeeze(0)

def custom_op_exec_tnd_pa( query, key, value, return_softmax_lse, block_table,
                          actseqlen, actseqlenkv, block_size):
    softmax_scale = 1 / 0.0078125
    return torch_npu.npu_fused_infer_attention_score_v2(
        query, key, value, num_query_heads=1, input_layout="TND", softmax_scale=softmax_scale,
        pre_tokens=65535, next_tokens=65535, return_softmax_lse=return_softmax_lse, block_table=block_table,
        actual_seq_qlen=actseqlen, actual_seq_kvlen=actseqlenkv, block_size=block_size)

def test_npu_fused_infer_attention_score_v2_tnd_pa( device="npu"):
    # T=128, N=1, D=128; 用 block_size=16,共 8 个物理块
    block_size = 16
    num_blocks = 8

    query = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
    key = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
    value = torch.full((128, 1, 128), 1, dtype=torch.bfloat16).npu()
    # block_table: (B=1, maxBlockNumPerSeq=8),顺序映射到第 0~7 块
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks).npu()

    return_softmax_lse = True

    actseqlen = [128]
    actseqlenkv = [128]

    golden_output = golden_op_exec(query, key, value)
    # KV cache 排布: (num_blocks, N, block_size, D) = (8, 1, 16, 128)
    # token 0~15 -> block0, 16~31 -> block1, ...
    key_cache = key.reshape(num_blocks, block_size, 1, 128).permute(0, 2, 1, 3).contiguous()
    value_cache = value.reshape(num_blocks, block_size, 1, 128).permute(0, 2, 1, 3).contiguous()

    custom_output = custom_op_exec_tnd_pa(query, key_cache, value_cache, return_softmax_lse, block_table,
                                          actseqlen, actseqlenkv, block_size)
    attention_output = custom_output[0]
    assert torch.allclose(golden_output, attention_output, rtol=1e-3, atol=1e-3), \
        f"golden={golden_output.flatten()[:8]}, actual={attention_output.flatten()[:8]}"
    print("\n[PASS] test_npu_fused_infer_attention_score_v2_tnd_pa")
    print(f"  golden  : {golden_output.flatten()[:8].tolist()}")
    print(f"  actual  : {attention_output.flatten()[:8].tolist()}")
    print(f"  max_diff: {(golden_output - attention_output).abs().max().item()}")


if __name__ == "__main__":
    test_npu_fused_infer_attention_score_v2_tnd_pa()