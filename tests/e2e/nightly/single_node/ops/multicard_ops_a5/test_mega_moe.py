import random
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401, E402

CORRECTNESS_BATCH_SIZES = (1, 4, 16, 64, 128)
E8M0_EXPONENT_BIAS = 127
E8M0_SCALE_ONE = 127
NON_UNIT_E8M0_SCALE_VALUES = (126, 127, 128)


def get_float8_e8m0_dtype():
    for module, name in (
        (torch, "float8_e8m0fnu"),
        (torch, "float8_e8m0"),
        (torch_npu, "float8_e8m0fnu"),
        (torch_npu, "float8_e8m0"),
    ):
        if hasattr(module, name):
            return getattr(module, name)
    raise RuntimeError("float8_e8m0 dtype is not available")


class TestMegaMoe:
    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port
        self.ep_group = None

    def get_hcomm(self, comm_group):
        return comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(
            self.rank)

    def generate_hcom(self):
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        self.ep_group = dist.new_group(backend="hccl",
                                       ranks=list(range(self.world_size)))
        self.hcomm_info = self.get_hcomm(self.ep_group)
        assert self.hcomm_info, "HCCL comm name is empty"

    @staticmethod
    def ceil(a, b):
        return (a + b - 1) // b

    @staticmethod
    def check_output(y, expert_token_nums, x, local_expert_num) -> bool:
        torch_npu.npu.synchronize()
        return (y.shape == x.shape and y.dtype == x.dtype
                and expert_token_nums.shape == (local_expert_num, )
                and expert_token_nums.dtype == torch.int32)

    @staticmethod
    def _reference_mega_moe(x, topk_ids, topk_weights, weight1, weight2,
                            moe_expert_num, ep_world_size,
                            weight_scales1=None, weight_scales2=None):
        expert_per_rank = moe_expert_num // ep_world_size
        x_fp32 = x.float()
        topk_weights_fp32 = topk_weights.float()
        topk_ids_cpu = topk_ids.cpu()
        weight1_fp32 = TestMegaMoe._dequant_mxfp8_weight(
            weight1[0], weight_scales1[0])
        weight2_fp32 = TestMegaMoe._dequant_mxfp8_weight(
            weight2[0], weight_scales2[0])

        ref = torch.zeros_like(x_fp32)
        for token_idx in range(x.shape[0]):
            token_out = torch.zeros_like(x_fp32[token_idx])
            for topk_idx in range(topk_ids.shape[1]):
                expert_id = int(topk_ids_cpu[token_idx, topk_idx])
                local_expert_id = expert_id % expert_per_rank
                hidden = torch.matmul(x_fp32[token_idx],
                                      weight1_fp32[local_expert_id].t())
                swish, gate = hidden.chunk(2, dim=-1)
                activated = F.silu(swish) * gate
                expert_out = torch.matmul(
                    activated, weight2_fp32[local_expert_id].t())
                token_out += topk_weights_fp32[token_idx,
                                               topk_idx] * expert_out
            ref[token_idx] = token_out
        return ref.to(x.dtype)

    @staticmethod
    def _e8m0_to_float(scale):
        scale_bits = scale.contiguous().view(torch.uint8).cpu().to(
            torch.float32)
        return torch.pow(torch.full_like(scale_bits, 2.0),
                         scale_bits - E8M0_EXPONENT_BIAS)

    @staticmethod
    def _dequant_mxfp8_weight(weight, scale):
        weight_fp32 = weight.float()
        scale_fp32 = TestMegaMoe._e8m0_to_float(scale).to(weight_fp32.device)
        scale_fp32 = scale_fp32.reshape(*scale_fp32.shape[:-2], -1)
        expanded_scale = scale_fp32.repeat_interleave(32, dim=-1)
        expanded_scale = expanded_scale[..., :weight_fp32.shape[-1]]
        return weight_fp32 * expanded_scale

    def _reference_expert_token_nums(self, topk_ids, local_expert_num):
        topk_ids_cpu = topk_ids.cpu()
        expert_start = self.rank * local_expert_num
        counts = torch.zeros(local_expert_num, dtype=torch.int32)
        for local_expert_id in range(local_expert_num):
            global_expert_id = expert_start + local_expert_id
            counts[local_expert_id] = int(
                (topk_ids_cpu == global_expert_id).sum()) * self.world_size
        return counts.npu()

    def _make_e8m0_scales(self, shape, use_non_unit_scale):
        fp8_e8m0 = get_float8_e8m0_dtype()
        if not use_non_unit_scale:
            scale = torch.full(shape, E8M0_SCALE_ONE, dtype=torch.uint8)
            return scale.view(fp8_e8m0).npu()

        scale = torch.empty(shape, dtype=torch.uint8)
        flat_scale = scale.view(-1)
        for offset, value in enumerate(NON_UNIT_E8M0_SCALE_VALUES):
            flat_scale[offset::len(NON_UNIT_E8M0_SCALE_VALUES)] = value
        return scale.view(fp8_e8m0).npu()

    def _make_correctness_inputs(self, weight_dtype, bs,
                                 use_non_unit_scale=False):
        torch.manual_seed(2026)
        h, topk, e, n = 4096, 6, 4, 1024
        n2 = n // 2

        x = (torch.randn(bs, h, dtype=torch.float32) * 0.1).to(
            torch.bfloat16).npu()
        topk_ids = torch.stack([
            torch.randperm(e * self.world_size)[:topk] for _ in range(bs)
        ]).to(torch.int32).npu()
        topk_weights = torch.full((bs, topk),
                                  1.0 / topk,
                                  dtype=torch.bfloat16).npu()

        weight1 = [(torch.randn(e, n, h, dtype=torch.float32) * 0.01).to(
            weight_dtype).npu()]
        weight2 = [(torch.randn(e, h, n2, dtype=torch.float32) * 0.01).to(
            weight_dtype).npu()]

        w1_scales = [
            self._make_e8m0_scales((e, n, self.ceil(h, 64), 2),
                                   use_non_unit_scale)
        ]
        w2_scales = [
            self._make_e8m0_scales((e, h, self.ceil(n2, 64), 2),
                                   use_non_unit_scale)
        ]
        return x, topk_ids, topk_weights, weight1, weight2, w1_scales, w2_scales

    def check_correctness(self, y, expert_token_nums, x, topk_ids,
                          topk_weights, weight1, weight2, weight_scales1,
                          weight_scales2, local_expert_num, moe_expert_num):
        torch_npu.npu.synchronize()
        y_ref = self._reference_mega_moe(x, topk_ids, topk_weights, weight1,
                                         weight2, moe_expert_num,
                                         self.world_size, weight_scales1,
                                         weight_scales2)
        expert_token_nums_ref = self._reference_expert_token_nums(
            topk_ids, local_expert_num)
        torch.testing.assert_close(expert_token_nums.cpu(),
                                   expert_token_nums_ref.cpu(),
                                   rtol=0,
                                   atol=0)
        torch.testing.assert_close(y.float().cpu(),
                                   y_ref.float().cpu(),
                                   rtol=5e-2,
                                   atol=5e-2)
        return True

    def run_correctness_fp8_e4m3(self) -> bool:
        torch_npu.npu.set_device(self.rank)

        for bs in CORRECTNESS_BATCH_SIZES:
            inputs = self._make_correctness_inputs(torch.float8_e4m3fn, bs)
            x, topk_ids, topk_weights, weight1, weight2, w1_scales, w2_scales = inputs
            local_expert_num = weight1[0].shape[0]
            moe_expert_num = local_expert_num * self.world_size

            y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
                x, topk_ids, topk_weights, weight1, weight2,
                group_ep=self.hcomm_info, moe_expert_num=moe_expert_num,
                weight_scales1=w1_scales, weight_scales2=w2_scales,
                dispatch_quant_mode=4, dispatch_quant_out_type=24)
            self.check_correctness(y, expert_token_nums, x, topk_ids,
                                   topk_weights, weight1, weight2, w1_scales,
                                   w2_scales,
                                   local_expert_num, moe_expert_num)
        inputs = self._make_correctness_inputs(torch.float8_e4m3fn,
                                               bs=4,
                                               use_non_unit_scale=True)
        x, topk_ids, topk_weights, weight1, weight2, w1_scales, w2_scales = inputs
        local_expert_num = weight1[0].shape[0]
        moe_expert_num = local_expert_num * self.world_size

        y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
            x, topk_ids, topk_weights, weight1, weight2,
            group_ep=self.hcomm_info, moe_expert_num=moe_expert_num,
            weight_scales1=w1_scales, weight_scales2=w2_scales,
            dispatch_quant_mode=4, dispatch_quant_out_type=24)
        self.check_correctness(y, expert_token_nums, x, topk_ids,
                               topk_weights, weight1, weight2, w1_scales,
                               w2_scales, local_expert_num, moe_expert_num)
        return True

    def run_correctness_fp8_e5m2(self) -> bool:
        torch_npu.npu.set_device(self.rank)

        for bs in CORRECTNESS_BATCH_SIZES:
            inputs = self._make_correctness_inputs(torch.float8_e5m2, bs)
            x, topk_ids, topk_weights, weight1, weight2, w1_scales, w2_scales = inputs
            local_expert_num = weight1[0].shape[0]
            moe_expert_num = local_expert_num * self.world_size

            y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
                x, topk_ids, topk_weights, weight1, weight2,
                group_ep=self.hcomm_info, moe_expert_num=moe_expert_num,
                weight_scales1=w1_scales, weight_scales2=w2_scales,
                dispatch_quant_mode=4, dispatch_quant_out_type=23)
            self.check_correctness(y, expert_token_nums, x, topk_ids,
                                   topk_weights, weight1, weight2, w1_scales,
                                   w2_scales,
                                   local_expert_num, moe_expert_num)
        inputs = self._make_correctness_inputs(torch.float8_e5m2,
                                               bs=4,
                                               use_non_unit_scale=True)
        x, topk_ids, topk_weights, weight1, weight2, w1_scales, w2_scales = inputs
        local_expert_num = weight1[0].shape[0]
        moe_expert_num = local_expert_num * self.world_size

        y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
            x, topk_ids, topk_weights, weight1, weight2,
            group_ep=self.hcomm_info, moe_expert_num=moe_expert_num,
            weight_scales1=w1_scales, weight_scales2=w2_scales,
            dispatch_quant_mode=4, dispatch_quant_out_type=23)
        self.check_correctness(y, expert_token_nums, x, topk_ids,
                               topk_weights, weight1, weight2, w1_scales,
                               w2_scales, local_expert_num, moe_expert_num)
        return True

    def run_forward_fp8_e4m3(self) -> bool:
        torch_npu.npu.set_device(self.rank)

        bs, h, topk, e, n = 128, 4096, 6, 4, 2048
        n2 = n // 2

        x = torch.randn(bs, h, dtype=torch.bfloat16).npu()
        topk_ids = torch.stack([
            torch.randperm(e * self.world_size)[:topk] for _ in range(bs)
        ]).to(torch.int32).npu()
        topk_weights = torch.randn(bs, topk, dtype=torch.bfloat16).npu()

        weight1 = [torch.randn(e, n, h, dtype=torch.float32).to(torch.float8_e4m3fn).npu()]
        weight2 = [torch.randn(e, h, n2, dtype=torch.float32).to(torch.float8_e4m3fn).npu()]

        fp8_e8m0 = get_float8_e8m0_dtype()
        w1_scales = [torch.randint(125, 130, (e, n, self.ceil(h, 64), 2),
                                   dtype=torch.uint8).view(fp8_e8m0).npu()]
        w2_scales = [torch.randint(125, 130, (e, h, self.ceil(n2, 64), 2),
                                   dtype=torch.uint8).view(fp8_e8m0).npu()]

        y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
            x, topk_ids, topk_weights, weight1, weight2,
            group_ep=self.hcomm_info, moe_expert_num=e * self.world_size,
            weight_scales1=w1_scales, weight_scales2=w2_scales,
            dispatch_quant_mode=4, dispatch_quant_out_type=24)
        return self.check_output(y, expert_token_nums, x, e)

    def run_forward_fp8_e5m2(self) -> bool:
        torch_npu.npu.set_device(self.rank)

        bs, h, topk, e, n = 256, 5120, 8, 8, 3072
        n2 = n // 2

        x = torch.randn(bs, h, dtype=torch.bfloat16).npu()
        topk_ids = torch.stack([
            torch.randperm(e * self.world_size)[:topk] for _ in range(bs)
        ]).to(torch.int32).npu()
        topk_weights = torch.randn(bs, topk, dtype=torch.bfloat16).npu()

        weight1 = [torch.randn(e, n, h, dtype=torch.float32).to(torch.float8_e5m2).npu()]
        weight2 = [torch.randn(e, h, n2, dtype=torch.float32).to(torch.float8_e5m2).npu()]

        fp8_e8m0 = get_float8_e8m0_dtype()
        w1_scales = [torch.randint(125, 130, (e, n, self.ceil(h, 64), 2),
                                   dtype=torch.uint8).view(fp8_e8m0).npu()]
        w2_scales = [torch.randint(125, 130, (e, h, self.ceil(n2, 64), 2),
                                   dtype=torch.uint8).view(fp8_e8m0).npu()]

        y, expert_token_nums = torch.ops._C_ascend.npu_mega_moe(
            x, topk_ids, topk_weights, weight1, weight2,
            group_ep=self.hcomm_info, moe_expert_num=e * self.world_size,
            weight_scales1=w1_scales, weight_scales2=w2_scales,
            dispatch_quant_mode=4, dispatch_quant_out_type=23)
        return self.check_output(y, expert_token_nums, x, e)


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    try:
        op = TestMegaMoe(rank, world_size, port)
        op.generate_hcom()
        q.put((rank, True, [
            op.run_forward_fp8_e4m3(),
            op.run_correctness_fp8_e4m3(),
        ], ""))
    except Exception:
        q.put((rank, False, [], traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@torch.inference_mode()
def test_mega_moe_fp8_e4m3():
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = ctx.Process(target=worker, args=(rank, world_size, port, q))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]

    for p in p_list:
        p.join()

    errors = [msg for msg in results if not msg[1] or not all(msg[2])]
    assert not errors, errors
    assert all(p.exitcode == 0 for p in p_list)


def worker1(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    try:
        op = TestMegaMoe(rank, world_size, port)
        op.generate_hcom()
        q.put((rank, True, [
            op.run_forward_fp8_e5m2(),
            op.run_correctness_fp8_e5m2(),
        ], ""))
    except Exception:
        q.put((rank, False, [], traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@torch.inference_mode()
def test_mega_moe_fp8_e5m2():
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = ctx.Process(target=worker1, args=(rank, world_size, port, q))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]

    for p in p_list:
        p.join()

    errors = [msg for msg in results if not msg[1] or not all(msg[2])]
    assert not errors, errors
    assert all(p.exitcode == 0 for p in p_list)
