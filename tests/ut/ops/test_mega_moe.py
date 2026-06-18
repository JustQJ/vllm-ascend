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
import os

import pytest
import torch
import torch_npu
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
import torch.distributed as dist

from vllm_ascend.ops.mega_moe import (
    SymmBuffer,
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
    """Verify the mega_moe API surface is importable and callable."""

    def test_import_available(self):
        assert callable(mega_moe)
        assert callable(get_symm_buffer_for_mega_moe)
        assert callable(npu_get_mega_moe_ccl_buffer_size)

    def test_ccl_buffer_size_returns_positive_int(self):
        size = npu_get_mega_moe_ccl_buffer_size(
            ep_world_size=2,
            moe_expert_num=8,
            num_max_tokens_per_rank=0,
            num_topk=6,
            hidden=4096,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=23,
        )
        assert isinstance(size, int), f"expected int, got {type(size)}"
        assert size > 0, f"expected positive CCL buffer size, got {size}"


# ---------------------------------------------------------------------------
# distributed integration test (requires NPU hardware + HCCL)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("RUN_MEGA_MOE_NPU_TEST") != "1",
    reason="Set RUN_MEGA_MOE_NPU_TEST=1 to run NPU distributed test",
)
class TestMegaMoeDistributed:
    """End-to-end multi-process test matching the ops-transformer mega_moe.md example.

    This test spawns WORLD_SIZE processes, each initializing HCCL communication,
    constructing a SymmBuffer, calling mega_moe(), and returning outputs via a
    shared queue.
    """

    # Hyper-parameters (same as ops-transformer mega_moe.md example)
    E = 4           # num_local_experts
    BS = 256        # batch size
    H = 4096        # hidden dim
    N = 1024        # intermediate dim
    TOPK = 6
    NUM_EXPERTS = 8
    WORLD_SIZE = 2

    @staticmethod
    def _run_megamoe_npu(
        queue, rank,
        x, topk_ids, topk_weights,
        w1, w1_scales, w2, w2_scales,
    ):
        """Single-rank process entry point."""
        torch_npu.npu.set_device(rank)

        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=TestMegaMoeDistributed.WORLD_SIZE,
            init_method="tcp://127.0.0.1:50001",
        )
        ep_group = dist.new_group(
            backend="hccl",
            ranks=list(range(TestMegaMoeDistributed.WORLD_SIZE)),
        )

        sym_buffer = get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=TestMegaMoeDistributed.NUM_EXPERTS,
            num_max_tokens_per_rank=0,
            num_topk=TestMegaMoeDistributed.TOPK,
            hidden=TestMegaMoeDistributed.H,
            intermediate_hidden=TestMegaMoeDistributed.N,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=23,  # FLOAT8_E5M2
        )

        y, expert_token_nums = mega_moe(
            sym_buffer=sym_buffer,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=[w1],
            l2_weights=[w2],
            l1_scales=[w1_scales] if w1_scales is not None else None,
            l2_scales=[w2_scales] if w2_scales is not None else None,
        )

        torch.npu.synchronize()
        dist.destroy_process_group()
        queue.put([rank, [y.cpu(), expert_token_nums.cpu()]])

    def test_mega_moe_end_to_end(self):
        E = self.E; BS = self.BS; H = self.H; N = self.N; TK = self.TOPK

        # Build inputs (bfloat16, non-quantized weights for basic sanity)
        x = torch.randn(BS, H, dtype=torch.bfloat16)
        topk_ids = torch.stack(
            [torch.randperm(self.NUM_EXPERTS)[:TK] for _ in range(BS)]
        ).to(torch.int32)
        topk_weights = torch.randn(BS, TK, dtype=torch.bfloat16)
        w1 = torch.randn(E, N, H, dtype=torch.float32)
        w2 = torch.randn(E, H, N // 2, dtype=torch.float32)

        manager = Manager()
        queue = manager.Queue()
        procs = []
        mp.set_start_method("forkserver", force=True)
        for rank in range(self.WORLD_SIZE):
            p = Process(
                target=self._run_megamoe_npu,
                args=(queue, rank, x, topk_ids, topk_weights,
                      w1, None, w2, None),
            )
            p.start()
            procs.append(p)

        outputs = [queue.get() for _ in procs]
        for p in procs:
            p.join()

        for rank, (y, expert_token_nums) in outputs:
            assert y.shape == (BS, H), \
                f"rank {rank}: expected y.shape ({BS}, {H}), got {y.shape}"
            assert expert_token_nums.shape == (E,), \
                f"rank {rank}: expected expert_token_nums.shape ({E},), got {expert_token_nums.shape}"

        print("mega_moe end-to-end test passed!")
