import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.ops.mega_moe import get_symm_buffer_for_mega_moe, mega_moe
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
import vllm_ascend.meta_registration  # type: ignore  # noqa: F401


def _ceil(a, b):
    return (a + b - 1) // b


class TestMegaMoe:
    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port

    def get_hcomm(self, comm_group):
        if torch.__version__ > "2.0.1":
            return comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        else:
            return comm_group.get_hccl_comm_name(self.rank)

    def generate_hcom(self):
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        if dist.is_available():
            default_pg = _get_default_group()
        self.hcomm_info = self.get_hcomm(default_pg)
        self.ep_group = dist.new_group(
            backend="hccl",
            ranks=list(range(self.world_size)),
        )

    # ------------------------------------------------------------------
    # non-quantized (float32 weights, no scales, no bias)
    # ------------------------------------------------------------------
    def run_no_quant_num_experts(self, num_experts: int, num_topk: int,
                                  bs: int, hidden: int, intermediate: int) -> bool:
        torch_npu.npu.set_device(self.rank)
        e = num_experts // self.world_size

        x = torch.randn(bs, hidden, dtype=torch.bfloat16).npu()
        topk_ids = torch.randint(0, num_experts, (bs, num_topk), dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, num_topk, dtype=torch.bfloat16).npu()
        w1 = torch.randn(e, intermediate, hidden, dtype=torch.float32).npu()
        w2 = torch.randn(e, hidden, intermediate * 2, dtype=torch.float32).npu()

        sym_buffer = get_symm_buffer_for_mega_moe(
            self.ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=0,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate,
            dispatch_quant_mode=0,
        )
        mega_moe(
            sym_buffer=sym_buffer,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=[w1],
            l2_weights=[w2],
        )
        return True

    def run_no_quant(self) -> bool:
        return self.run_no_quant_num_experts(
            num_experts=8, num_topk=6, bs=256, hidden=4096, intermediate=1024)

    # ------------------------------------------------------------------
    # MXFP quantized (float8_e5m2 weights + float8_e8m0 scales)
    # ------------------------------------------------------------------
    def run_quant(self) -> bool:
        if not hasattr(torch, "float8_e5m2"):
            return True

        torch_npu.npu.set_device(self.rank)
        num_experts = 8
        num_topk = 6
        bs = 256
        hidden = 4096
        intermediate = 1024
        e = num_experts // self.world_size

        x = torch.randn(bs, hidden, dtype=torch.bfloat16).npu()
        topk_ids = torch.randint(0, num_experts, (bs, num_topk), dtype=torch.int32).npu()
        topk_weights = torch.randn(bs, num_topk, dtype=torch.bfloat16).npu()

        w1 = torch.randn(e, intermediate, hidden, dtype=torch.float32).to(torch.float8_e5m2).npu()
        w1_scale_shape = (e, intermediate, _ceil(hidden, 64), 2)
        w1_scales = (
            torch.randint(125, 130, w1_scale_shape, dtype=torch.uint8)
            .view(torch.float8_e8m0fnu).npu()
        )

        w2 = torch.randn(e, hidden, intermediate * 2, dtype=torch.float32).to(torch.float8_e5m2).npu()
        w2_scale_shape = (e, hidden, _ceil(intermediate * 2, 64), 2)
        w2_scales = (
            torch.randint(125, 130, w2_scale_shape, dtype=torch.uint8)
            .view(torch.float8_e8m0fnu).npu()
        )

        sym_buffer = get_symm_buffer_for_mega_moe(
            self.ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=0,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=23,  # FLOAT8_E5M2
        )
        mega_moe(
            sym_buffer=sym_buffer,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=[w1],
            l2_weights=[w2],
            l1_weights_sf=[w1_scales],
            l2_weights_sf=[w2_scales],
        )
        return True


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    op = TestMegaMoe(rank, world_size, port)
    op.generate_hcom()
    out1 = op.run_no_quant()
    q.put(out1)
    out2 = op.run_quant()
    q.put(out2)


@torch.inference_mode()
def test_mega_moe_kernel():
    world_size = 2
    mp.set_start_method("fork", force=True)

    q = mp.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, port, q))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size * 2)]

    for p in p_list:
        p.join()

    assert all(results)
