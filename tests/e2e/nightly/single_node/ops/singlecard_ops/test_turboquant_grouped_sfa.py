# SPDX-License-Identifier: Apache-2.0
"""On-device acceptance tests for the opt-in Scheme A prototype.

These must pass on A2/A3 before switching the framework's default entry point.
"""

import math

import numpy as np
import pytest
import torch
import torch_npu

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_turboquant_custom_ops import (
    TQ_BLOCK_SIZE,
    TQ_HEAD_DIM,
    TQ_ROPE_HEAD_DIM,
    TQ_SLOT_BYTES,
    _pack_sfa_slots,
)

HEADS = 16
CONTEXT = 16384


def _inputs(lengths, topk=512, selection="overlap", heads=HEADS, sparse_mode=3):
    rng = np.random.default_rng(20260909)
    tokens = sum(lengths)
    # Nonzero RoPE and nonunit, varying scales exercise column alignment.
    nope = rng.standard_normal((CONTEXT, TQ_HEAD_DIM)).astype(np.float32)
    nope *= rng.uniform(0.5, 2.0, (CONTEXT, 1)).astype(np.float32)
    rope = rng.standard_normal((CONTEXT, TQ_ROPE_HEAD_DIM)).astype(np.float32)
    slot, unit, scales, stored_rope = _pack_sfa_slots(nope, rope)
    blocks = CONTEXT // TQ_BLOCK_SIZE
    table = np.stack([rng.permutation(blocks) for _ in lengths]).astype(np.int32)
    query = torch.from_numpy(rng.standard_normal((tokens, heads, 576)).astype(np.float32)).to(torch.bfloat16)
    indices = np.empty((tokens, 1, topk), dtype=np.int32)
    first = 0
    for length in lengths:
        for q in range(length):
            if selection == "disjoint":
                start = (q % 4) * topk
            elif selection == "pairs":
                start = (q % 4 // 2) * (2 * topk)
            else:
                start = q % 4 * 3
            indices[first + q, 0] = rng.permutation(np.arange(start, start + topk, dtype=np.int32))
            if selection == "duplicates":
                indices[first + q, 0, 1] = indices[first + q, 0, 0]
        first += length
    args = (
        query.npu(),
        torch.from_numpy(slot.view(np.int8)).reshape(blocks, TQ_BLOCK_SIZE, 1, TQ_SLOT_BYTES).npu(),
        torch.from_numpy(indices).npu(),
    )
    kwargs = {
        "block_table": torch.from_numpy(table).npu(),
        "actual_seq_lengths_query": torch.tensor(np.cumsum(lengths), dtype=torch.int32).npu(),
        "actual_seq_lengths_kv": torch.full((len(lengths),), CONTEXT, dtype=torch.int32).npu(),
        "scale_value": 1.0 / math.sqrt(TQ_HEAD_DIM),
        "key_quant_mode": 3,
        "value_quant_mode": 3,
        "sparse_block_size": 1,
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": sparse_mode,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": TQ_BLOCK_SIZE,
        "rope_head_dim": TQ_ROPE_HEAD_DIM,
        "return_softmax_lse": True,
    }
    # The device byte-LUT rounds the unit vector before MM1/MM2.
    unit = torch.from_numpy(unit).to(torch.bfloat16).double()
    return args, kwargs, (unit, torch.from_numpy(scales.astype(np.float64)), torch.from_numpy(stored_rope))


def _call(name, args, kwargs):
    query, kv, indices = args
    return getattr(torch.ops._C_ascend, name)(query, kv, kv, indices, **kwargs)


def _reference(args, kwargs, cache):
    unit, scales, rope = cache
    query, _, indices = args
    query = query.cpu().double()
    indices = indices.cpu().numpy()
    table = kwargs["block_table"].cpu().numpy()
    cumulative = kwargs["actual_seq_lengths_query"].cpu().tolist()
    kv_lengths = kwargs["actual_seq_lengths_kv"].cpu().tolist()
    out = torch.zeros(query.shape[0], query.shape[1], TQ_HEAD_DIM, dtype=torch.float64)
    lse = torch.full(query.shape[:2], -torch.inf, dtype=torch.float64)
    first = 0
    for batch, end in enumerate(cumulative):
        for t in range(first, end):
            ids = indices[t, 0]
            limit = kv_lengths[batch]
            if kwargs["sparse_mode"] == 3:
                limit = min(limit, kv_lengths[batch] - (end - first) + t - first + 1)
            ids = ids[(ids >= 0) & (ids < limit)]
            if ids.size == 0:
                continue
            physical = table[batch, ids // TQ_BLOCK_SIZE].astype(np.int64) * TQ_BLOCK_SIZE + ids % TQ_BLOCK_SIZE
            c, s, r = unit[physical], scales[physical], rope[physical]
            score = (query[t, :, :TQ_HEAD_DIM] @ c.T + query[t, :, TQ_HEAD_DIM:] @ r.T)
            score *= s[None, :] * kwargs["scale_value"]
            out[t] = torch.softmax(score, dim=-1) @ (c * s[:, None])
            lse[t] = torch.logsumexp(score, dim=-1)
        first = end
    return out, lse


def _assert_reference(actual, expected):
    output, maximum, total = actual
    ref_output, ref_lse = expected
    torch.testing.assert_close(output.cpu().double(), ref_output, atol=0.015, rtol=0.025)
    lse = (maximum.float() + total.float().log()).squeeze(0).cpu().double()
    torch.testing.assert_close(lse, ref_lse, atol=0.025, rtol=0.005)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("lengths", "topk", "selection"),
    [
        ([1], 128, "overlap"),
        ([2], 513, "overlap"),
        ([3], 512, "disjoint"),
        ([5, 0, 3, 1], 512, "disjoint"),
        ([4], 2048, "pairs"),
        ([4], 2048, "disjoint"),
        ([7], 128, "duplicates"),
    ],
)
def test_grouped_sfa_reference(lengths, topk, selection):
    args, kwargs, cache = _inputs(lengths, topk, selection)
    original_indices = args[2].clone()
    actual = _call("turboquant_sparse_flash_attention_grouped", args, kwargs)
    _assert_reference(actual, _reference(args, kwargs, cache))
    torch.testing.assert_close(args[2], original_indices, rtol=0, atol=0)


@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_grouped_sfa_causal_and_empty_membership(sparse_mode):
    args, kwargs, cache = _inputs([4], 512, "disjoint", sparse_mode=sparse_mode)
    indices = args[2].cpu()
    indices[0].fill_(-1)  # one query has no membership in any union tile
    indices[1, 0, -3:] = torch.tensor([CONTEXT - 1, CONTEXT - 2, CONTEXT - 3])
    args[2].copy_(indices)
    _assert_reference(
        _call("turboquant_sparse_flash_attention_grouped", args, kwargs),
        _reference(args, kwargs, cache),
    )
    kwargs["actual_seq_lengths_kv"].zero_()
    _assert_reference(
        _call("turboquant_sparse_flash_attention_grouped", args, kwargs),
        _reference(args, kwargs, cache),
    )


def test_grouped_sfa_unsupported_heads_use_legacy():
    args, kwargs, _ = _inputs([3], 128, heads=8)
    actual = _call("turboquant_sparse_flash_attention_grouped", args, kwargs)
    expected = _call("turboquant_sparse_flash_attention", args, kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_grouped_sfa_graph_replay_changes_lengths_and_fallback():
    args, kwargs, cache = _inputs([4, 4], 2048)
    stream = torch_npu.npu.Stream()
    stream.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(stream):
        for _ in range(3):
            _call("turboquant_sparse_flash_attention_grouped", args, kwargs)
    torch_npu.npu.current_stream().wait_stream(stream)
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph, stream=stream):
        actual = _call("turboquant_sparse_flash_attention_grouped", args, kwargs)
    # The same tensor addresses and captured two-op sequence are reused.
    scenarios = [
        ([4, 4], "overlap"),
        ([3, 5], "pairs"),
        ([1, 7], "disjoint"),
        ([5, 3], "duplicates"),
        ([3, 0], "overlap"),  # capacity remains 8; trailing rows must be zero
        ([0, 0], "overlap"),
    ]
    for lengths, selection in scenarios * 2:
        indices = args[2].cpu()
        first = 0
        for length in lengths:
            for q in range(length):
                if selection == "disjoint":
                    start = (q % 4) * 2048
                elif selection == "pairs":
                    start = (q % 4 // 2) * 4096
                else:
                    start = 0
                indices[first + q, 0] = torch.arange(start, start + 2048, dtype=torch.int32)
                if selection == "duplicates":
                    indices[first + q, 0, 1] = indices[first + q, 0, 0]
            first += length
        args[2].copy_(indices)
        kwargs["actual_seq_lengths_query"].copy_(torch.tensor(np.cumsum(lengths), dtype=torch.int32))
        graph.replay()
        _assert_reference(actual, _reference(args, kwargs, cache))
