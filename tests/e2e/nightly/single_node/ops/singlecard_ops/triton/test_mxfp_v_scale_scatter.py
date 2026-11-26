# SPDX-License-Identifier: Apache-2.0
"""Shared-pool V scale writes preserve other owners, components, and padding."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_c4_mxfp import mxfp4_packet_section_sizes, mxfp4_paged_cache_views
from vllm_ascend.attention.attention_c8_mxfp import mxfp_packet_size_bytes, mxfp_paged_cache_views
from vllm_ascend.ops.triton.mxfp_v_scale import scatter_v_scales


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("heads,dim", [(1, 128), (1, 256), (2, 256)])
@pytest.mark.parametrize("padding", [0, 4096])
def test_shared_owners_recycle_and_graph(bits, heads, dim, padding):
    blocks, page, offset = 8, 512, 4096
    payload = (
        sum(mxfp4_packet_section_sizes(heads, page, dim, dim))
        if bits == 4
        else mxfp_packet_size_bytes(heads, page, dim, dim)
    )
    pitch = payload + padding
    factory = mxfp4_paged_cache_views if bits == 4 else mxfp_paged_cache_views
    raw = torch.full((offset + blocks * pitch,), 13, dtype=torch.int8, device="npu")
    cache = factory(raw[offset:], blocks, heads, dim, dim, page, page_stride_bytes=pitch)[3]
    expected = raw.cpu()
    ref = factory(expected[offset:], blocks, heads, dim, dim, page, page_stride_bytes=pitch)[3]
    scale_a = (torch.arange(heads * dim) % 5 + 120).to(torch.uint8)
    scale_b = (torch.arange(heads * dim) % 7 + 115).to(torch.uint8)
    slots_a = torch.tensor([519, 520, 543, 544, 1023, -1, 0, 1569], dtype=torch.int64)
    slots_b = torch.tensor([1031, 1032, 1055, 1056, 1535, -1, -1, 2081], dtype=torch.int64)
    a, b = slots_a.npu(), slots_b.npu()
    sa, sb = scale_a.npu(), scale_b.npu()

    def reference(slots, scale):
        for slot in slots.tolist():
            if slot < 0:
                continue
            block, token = divmod(slot, page)
            if bits == 4:
                ref[block, token // 64, :, :, :] = scale.reshape(heads, dim, 1)
            else:
                ref[block, :, :, token // 64, :, :] = scale.reshape(heads, dim // 16, 16, 1)

    def invoke():
        scatter_v_scales(sa, cache, a)
        scatter_v_scales(sb, cache, b)

    invoke()
    reference(slots_a, scale_a)
    reference(slots_b, scale_b)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke()
    # Fresh owner assignments, changed scale contents, and a padded token.
    slots_a = torch.where(slots_a >= 0, slots_a + 1536, slots_a)
    slots_b = torch.where(slots_b >= 0, slots_b + 1536, slots_b)
    a.copy_(slots_a.npu())
    b.copy_(slots_b.npu())
    scale_a = scale_a + 1
    sa.copy_(scale_a.npu())
    graph.replay()
    reference(slots_a, scale_a)
    reference(slots_b, scale_b)
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, atol=0, rtol=0)
    # Recycle A's slots to B. Remaining cache bytes must remain unchanged.
    scatter_v_scales(sb, cache, a)
    reference(slots_a, scale_b)
    scatter_v_scales(sa, cache, a[:0])
    torch.npu.synchronize()
    torch.testing.assert_close(raw.cpu(), expected, atol=0, rtol=0)
