import unittest

import torch
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()

# register custom ops into torch_library here
import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

# # register the meta implementation for custom kernel if necessary
# import vllm_ascend.meta_registration  # type: ignore  # noqa: F401


def _dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _write_norm_cache(cache, slot, token):
    block_size = cache.shape[1]
    cache[slot // block_size, slot % block_size, :, :] = token


def _write_compress_cache(cache, slot, token):
    block_size = cache.shape[1]
    cache[slot // block_size, slot % block_size, 0, :] = token


def _reference_normal_cache(key, key_cache, slot_mapping, value, value_cache):
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    key_cpu = key.cpu()
    value_cpu = value.cpu()

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        _write_norm_cache(expected_key_cache, slot, key_cpu[token_idx])
        if value_cpu.numel() > 0:
            _write_norm_cache(expected_value_cache, slot, value_cpu[token_idx])

    return expected_key_cache, expected_value_cache


def _reference_nct_cache(key_storage, key_cache, key_stride, key_offset,
                         slot_mapping, value_storage, value_cache,
                         value_stride, value_offset, num_heads, k_head_size,
                         v_head_size):
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    key_storage_cpu = key_storage.cpu()
    value_storage_cpu = value_storage.cpu()

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        key_start = key_offset + token_idx * key_stride
        value_start = value_offset + token_idx * value_stride
        key_token = key_storage_cpu[key_start:key_start +
                                    num_heads * k_head_size]
        value_token = value_storage_cpu[value_start:value_start +
                                        num_heads * v_head_size]
        _write_norm_cache(
            expected_key_cache, slot, key_token.view(num_heads, k_head_size))
        _write_norm_cache(
            expected_value_cache, slot,
            value_token.view(num_heads, v_head_size))

    return expected_key_cache, expected_value_cache


def _reference_pa_nz_cache(key, key_cache, slot_mapping, value, value_cache):
    block_size = key_cache.shape[-2]
    last_dim_k = 32 // _dtype_size(key_cache.dtype)
    last_dim_v = 32 // _dtype_size(value_cache.dtype)
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    key_cpu = key.cpu()
    value_cpu = value.cpu()

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        block_index = slot // block_size
        block_offset = slot % block_size

        token_key = key_cpu[token_idx].reshape(-1)
        token_value = value_cpu[token_idx].reshape(-1)
        for key_idx in range(token_key.numel() // last_dim_k):
            expected_key_cache[block_index, key_idx, block_offset, :] = (
                token_key[key_idx * last_dim_k:(key_idx + 1) * last_dim_k])
        for value_idx in range(token_value.numel() // last_dim_v):
            expected_value_cache[block_index, value_idx, block_offset, :] = (
                token_value[value_idx * last_dim_v:(value_idx + 1) * last_dim_v])

    return expected_key_cache, expected_value_cache


def _batch_start_offsets(seq_lens):
    starts = []
    offset = 0
    for seq_len in seq_lens:
        starts.append(offset)
        offset += seq_len
    return starts


def _reference_alibi_cache(key, key_cache, slot_mapping, value, value_cache,
                           compress_lens, seq_lens):
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    key_cpu = key.cpu()
    value_cpu = value.cpu()
    slot_mapping_cpu = slot_mapping.cpu().tolist()
    compress_lens_cpu = compress_lens.cpu().tolist()
    seq_lens_cpu = seq_lens.cpu().tolist()
    num_heads = key.shape[1]

    if key_cpu.dim() == 4:
        num_heads = key.shape[2]
        for batch_idx, seq_len in enumerate(seq_lens_cpu):
            for head_idx in range(num_heads):
                head_win = compress_lens_cpu[batch_idx][head_idx]
                slot = slot_mapping_cpu[batch_idx][head_idx]
                for win_idx in range(head_win):
                    token_offset = seq_len - head_win + win_idx
                    _write_compress_cache(
                        expected_key_cache, slot + win_idx,
                        key_cpu[batch_idx, token_offset, head_idx])
                    _write_compress_cache(
                        expected_value_cache, slot + win_idx,
                        value_cpu[batch_idx, token_offset, head_idx])
    else:
        batch_starts = _batch_start_offsets(seq_lens_cpu)
        for batch_idx, seq_len in enumerate(seq_lens_cpu):
            batch_end = batch_starts[batch_idx] + seq_len
            for head_idx in range(num_heads):
                task_idx = batch_idx * num_heads + head_idx
                head_win = compress_lens_cpu[task_idx]
                slot = slot_mapping_cpu[task_idx]
                for win_idx in range(head_win):
                    token_idx = batch_end - head_win + win_idx
                    _write_compress_cache(expected_key_cache, slot + win_idx,
                                          key_cpu[token_idx, head_idx])
                    _write_compress_cache(expected_value_cache, slot + win_idx,
                                          value_cpu[token_idx, head_idx])

    return expected_key_cache, expected_value_cache


def _reference_rope_or_omni_cache(key, key_cache, slot_mapping, value,
                                  value_cache, compress_lens,
                                  compress_seq_offset, seq_lens,
                                  scatter_mode):
    expected_key_cache = key_cache.cpu().clone()
    expected_value_cache = value_cache.cpu().clone()
    key_cpu = key.cpu()
    value_cpu = value.cpu()
    slot_mapping_cpu = slot_mapping.cpu().tolist()
    compress_lens_cpu = compress_lens.cpu().tolist()
    offsets_cpu = compress_seq_offset.cpu().tolist()
    seq_lens_cpu = seq_lens.cpu().tolist()
    num_heads = key.shape[1]

    if key_cpu.dim() == 4:
        num_heads = key.shape[2]

        def get_token(batch_idx, token_offset, head_idx):
            return key_cpu[batch_idx, token_offset, head_idx]

        def get_value_token(batch_idx, token_offset, head_idx):
            return value_cpu[batch_idx, token_offset, head_idx]

        def get_task_params(batch_idx, head_idx):
            task_idx = batch_idx * num_heads + head_idx
            return (slot_mapping_cpu[batch_idx][head_idx],
                    compress_lens_cpu[batch_idx][head_idx],
                    offsets_cpu[task_idx])
    else:
        batch_starts = _batch_start_offsets(seq_lens_cpu)

        def get_token(batch_idx, token_offset, head_idx):
            return key_cpu[batch_starts[batch_idx] + token_offset, head_idx]

        def get_value_token(batch_idx, token_offset, head_idx):
            return value_cpu[batch_starts[batch_idx] + token_offset, head_idx]

        def get_task_params(batch_idx, head_idx):
            task_idx = batch_idx * num_heads + head_idx
            return (slot_mapping_cpu[task_idx], compress_lens_cpu[task_idx],
                    offsets_cpu[task_idx])

    for batch_idx, seq_len in enumerate(seq_lens_cpu):
        for head_idx in range(num_heads):
            slot, head_win, offset = get_task_params(batch_idx, head_idx)
            if offset == -1 or head_win == 0:
                offset = seq_len
                head_win = 0

            for token_offset in range(offset):
                _write_compress_cache(expected_key_cache, slot + token_offset,
                                      get_token(batch_idx, token_offset,
                                                head_idx))
                _write_compress_cache(expected_value_cache, slot + token_offset,
                                      get_value_token(batch_idx, token_offset,
                                                      head_idx))

            if scatter_mode == "Rope" and head_win != 0:
                key_avg = torch.stack([
                    get_token(batch_idx, idx, head_idx)
                    for idx in range(offset, offset + head_win)
                ]).to(torch.float32).mean(dim=0).to(key_cache.dtype)
                value_avg = torch.stack([
                    get_value_token(batch_idx, idx, head_idx)
                    for idx in range(offset, offset + head_win)
                ]).to(torch.float32).mean(dim=0).to(value_cache.dtype)
                _write_compress_cache(expected_key_cache, slot + offset,
                                      key_avg)
                _write_compress_cache(expected_value_cache, slot + offset,
                                      value_avg)

            if head_win != 0:
                dst_offset = offset + (1 if scatter_mode == "Rope" else 0)
                src_offset = offset + head_win
                for token_offset in range(src_offset, seq_len):
                    dst_slot = slot + dst_offset + token_offset - src_offset
                    _write_compress_cache(expected_key_cache, dst_slot,
                                          get_token(batch_idx, token_offset,
                                                    head_idx))
                    _write_compress_cache(expected_value_cache, dst_slot,
                                          get_value_token(batch_idx,
                                                          token_offset,
                                                          head_idx))

    return expected_key_cache, expected_value_cache


class TestScatterPaKvCache(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("NPU is required for npu_scatter_pa_kv_cache")
        torch.npu.set_device(0)

    def _run_pa_nz_case(self, *, slot_dtype, full_load):
        num_heads = 1
        block_size = 16 if full_load else 4
        num_blocks = 20 if full_load else 4
        token_num = 256 if full_load else 8
        k_head_size = 512 if full_load else 131072
        v_head_size = 64
        last_dim_k = 32 // 2
        last_dim_v = 32 // 2

        key = torch.randn(token_num, num_heads, k_head_size,
                          dtype=torch.float16).npu()
        value = torch.randn(token_num, num_heads, v_head_size,
                            dtype=torch.float16).npu()
        key_cache = torch.randn(
            num_blocks, num_heads * k_head_size // last_dim_k,
            block_size, last_dim_k, dtype=torch.float16).npu()
        value_cache = torch.randn(
            num_blocks, num_heads * v_head_size // last_dim_v,
            block_size, last_dim_v, dtype=torch.float16).npu()

        slots = torch.arange(token_num - 1, -1, -1, dtype=slot_dtype)
        slot_mapping = slots.npu()
        expected_key_cache, expected_value_cache = _reference_pa_nz_cache(
            key, key_cache, slot_mapping, value, value_cache)

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            None, None, None,
            "PA_NZ", "None",
            None, None)

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def test_scenario_1_pa_nz_full_load_int32(self):
        self._run_pa_nz_case(slot_dtype=torch.int32, full_load=True)

    def test_scenario_1_pa_nz_full_load_int64(self):
        self._run_pa_nz_case(slot_dtype=torch.int64, full_load=True)

    def test_scenario_1_pa_nz_not_full_load_int32(self):
        self._run_pa_nz_case(slot_dtype=torch.int32, full_load=False)

    def test_scenario_1_pa_nz_not_full_load_int64(self):
        self._run_pa_nz_case(slot_dtype=torch.int64, full_load=False)

    def test_scenario_2_normal_cache(self):
        num_blocks = 4
        num_heads = 2
        block_size = 4
        k_head_size = 64
        v_head_size = 128
        token_num = 8

        key = torch.randn(token_num, num_heads, k_head_size,
                          dtype=torch.float16).npu()
        value = torch.randn(token_num, num_heads, v_head_size,
                            dtype=torch.float16).npu()
        key_cache = torch.randn(
            num_blocks, block_size, num_heads, k_head_size,
            dtype=torch.float16).npu()
        value_cache = torch.randn(
            num_blocks, block_size, num_heads, v_head_size,
            dtype=torch.float16).npu()
        slot_mapping = torch.tensor(
            [7, 2, 0, 11, 5, -1, 3, 12], dtype=torch.int32).npu()
        expected_key_cache, expected_value_cache = _reference_normal_cache(
            key, key_cache, slot_mapping, value, value_cache)

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            None, None, None,
            "Norm", "None",
            None, None)

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def test_scenario_2_normal_cache_bf16(self):
        num_blocks = 2
        num_heads = 2
        block_size = 4
        head_size = 64
        token_num = 4

        key = torch.randn(token_num, num_heads, head_size,
                          dtype=torch.bfloat16).npu()
        value = torch.randn(token_num, num_heads, head_size,
                            dtype=torch.bfloat16).npu()
        key_cache = torch.randn(
            num_blocks, block_size, num_heads, head_size,
            dtype=torch.bfloat16).npu()
        value_cache = torch.randn(
            num_blocks, block_size, num_heads, head_size,
            dtype=torch.bfloat16).npu()
        slot_mapping = torch.tensor([7, 2, -1, 4], dtype=torch.int32).npu()
        expected_key_cache, expected_value_cache = _reference_normal_cache(
            key, key_cache, slot_mapping, value, value_cache)

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            None, None, None,
            "Norm", "None",
            None, None)

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def test_scenario_2_normal_cache_nct_strides_offsets(self):
        num_blocks = 4
        block_size = 4
        num_heads = 2
        k_head_size = 16
        v_head_size = 32
        token_num = 4
        key_token_size = num_heads * k_head_size
        value_token_size = num_heads * v_head_size
        key_stride = 20
        value_stride = 45
        key_offset = 10
        value_offset = 10

        key = torch.arange(
            token_num * key_token_size, dtype=torch.float16).view(
                token_num, num_heads, k_head_size).npu()
        value = (torch.arange(
            token_num * value_token_size, dtype=torch.float16) + 1000).view(
                token_num, num_heads, v_head_size).npu()
        key_cache = torch.randn(
            num_blocks, block_size, num_heads, k_head_size,
            dtype=torch.float16).npu()
        value_cache = torch.randn(
            num_blocks, block_size, num_heads, v_head_size,
            dtype=torch.float16).npu()
        slot_mapping = torch.tensor([7, 2, -1, 4], dtype=torch.int32).npu()
        expected_key_cache, expected_value_cache = _reference_nct_cache(
            key.reshape(-1), key_cache, key_stride, key_offset,
            slot_mapping, value.reshape(-1), value_cache, value_stride,
            value_offset, num_heads, k_head_size, v_head_size)

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            None, None, None,
            "Norm", "Nct",
            [key_stride, value_stride], [key_offset, value_offset])

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def test_scenario_4_alibi_compress(self):
        num_blocks = 4
        block_size = 4
        batch = 2
        seq_len = 4
        num_heads = 2
        head_size = 16
        seq_lens = torch.tensor([4, 3], dtype=torch.int32).npu()

        key = torch.randn(batch, seq_len, num_heads, head_size,
                          dtype=torch.float16).npu()
        value = torch.randn(batch, seq_len, num_heads, head_size,
                            dtype=torch.float16).npu()
        key_cache = torch.randn(
            num_blocks, block_size, 1, head_size, dtype=torch.float16).npu()
        value_cache = torch.randn(
            num_blocks, block_size, 1, head_size, dtype=torch.float16).npu()
        slot_mapping = torch.tensor([[0, 3], [6, 10]],
                                    dtype=torch.int32).npu()
        compress_lens = torch.tensor([[2, 1], [3, 2]],
                                     dtype=torch.int32).npu()
        expected_key_cache, expected_value_cache = _reference_alibi_cache(
            key, key_cache, slot_mapping, value, value_cache,
            compress_lens, seq_lens)

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            compress_lens, None, seq_lens,
            "Norm", "Alibi",
            None, None)

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def _run_scenario_5_compress_case(self, scatter_mode):
        num_blocks = 4
        block_size = 4
        batch = 2
        seq_len = 4
        num_heads = 2
        head_size = 16
        seq_lens = torch.tensor([4, 4], dtype=torch.int32).npu()

        key = torch.zeros(batch, seq_len, num_heads, head_size,
                          dtype=torch.float16)
        value = torch.zeros(batch, seq_len, num_heads, head_size,
                            dtype=torch.float16)
        for batch_idx in range(batch):
            for token_idx in range(seq_len):
                for head_idx in range(num_heads):
                    base = float((batch_idx + 1) * 100 +
                                 (token_idx + 1) * 10 + head_idx)
                    key[batch_idx, token_idx, head_idx, :] = base
                    value[batch_idx, token_idx, head_idx, :] = base + 1000
        key = key.npu()
        value = value.npu()

        key_cache = torch.randn(
            num_blocks, block_size, 1, head_size, dtype=torch.float16).npu()
        value_cache = torch.randn(
            num_blocks, block_size, 1, head_size, dtype=torch.float16).npu()
        slot_mapping = torch.tensor([[0, 4], [8, 12]],
                                    dtype=torch.int32).npu()
        compress_lens = torch.tensor([[3, 3], [3, 3]],
                                     dtype=torch.int32).npu()
        compress_seq_offset = torch.tensor([1, 1, 1, 1],
                                           dtype=torch.int32).npu()
        expected_key_cache, expected_value_cache = (
            _reference_rope_or_omni_cache(
                key, key_cache, slot_mapping, value, value_cache,
                compress_lens, compress_seq_offset, seq_lens, scatter_mode))

        torch.ops._C_ascend.npu_scatter_pa_kv_cache(
            key, key_cache, slot_mapping, value, value_cache,
            compress_lens, compress_seq_offset, seq_lens,
            "Norm", scatter_mode,
            None, None)

        torch.testing.assert_close(
            key_cache.cpu(), expected_key_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            value_cache.cpu(), expected_value_cache, rtol=0, atol=0)

    def test_scenario_5_rope_compress(self):
        self._run_scenario_5_compress_case("Rope")

    def test_scenario_5_omni_compress(self):
        self._run_scenario_5_compress_case("Omni")


if __name__ == "__main__":
    unittest.main()
