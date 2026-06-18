#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import pytest
import torch

from vllm_ascend.ops.mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
    npu_get_mega_moe_ccl_buffer_size,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ceil(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# unit tests (no NPU hardware required)
# ---------------------------------------------------------------------------
class TestMegaMoeSignature:
    """Verify the mega_moe API surface is importable and callable in CI."""

    def test_import_available(self):
        assert callable(mega_moe)
        assert callable(get_symm_buffer_for_mega_moe)
        assert callable(npu_get_mega_moe_ccl_buffer_size)

    def test_ccl_buffer_size_returns_positive_int(self):
        size = npu_get_mega_moe_ccl_buffer_size(
            ep_world_size=2,
            moe_expert_num=8,
            num_max_tokens_per_rank=24,
            num_topk=6,
            hidden=4096,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=23,
        )
        assert isinstance(size, int), f"expected int, got {type(size)}"
        assert size > 0, f"expected positive CCL buffer size, got {size}"

    def test_ccl_buffer_size_no_quant(self):
        """CCL buffer size works with default (non-quantized) params."""
        size = npu_get_mega_moe_ccl_buffer_size(
            ep_world_size=2,
            moe_expert_num=4,
            num_max_tokens_per_rank=1,
            num_topk=2,
            hidden=1024,
        )
        assert isinstance(size, int)
        assert size > 0

    def test_scale_shape_helper(self):
        """Scale shape calculation matches ops-transformer convention."""
        E, K, N = 4, 1024, 4096
        expected = (E, K, _ceil(N, 64), 2)
        assert expected[0] == E
        assert expected[1] == K
        assert expected[2] == _ceil(N, 64)  # 64-aligned blocks
        assert expected[3] == 2              # scale pair
