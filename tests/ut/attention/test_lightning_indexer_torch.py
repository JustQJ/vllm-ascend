# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.lightning_indexer_torch import (
    pytorch_lightning_indexer,
    pytorch_quant_lightning_indexer,
)


def test_pytorch_lightning_indexer_tnd_pa_scores_and_topk():
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=torch.float16,
    )
    weights = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float16)
    key = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 1.0]], [[2.0, 0.0]]],
        ],
        dtype=torch.float16,
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    topk_indices, score = pytorch_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([4], dtype=torch.int32),
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2,
        sparse_mode=0,
    )

    expected_score = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 2.0]],
            [[-0.5, 1.0, 0.0, -1.0]],
        ]
    )
    assert torch.allclose(score[:, 0], expected_score)
    assert torch.equal(topk_indices[:, 0], torch.tensor([[2, 1], [1, 2]], dtype=torch.int32))


def test_pytorch_lightning_indexer_causal_short_visible_len():
    query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float16)
    weights = torch.ones((2, 1), dtype=torch.float16)
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float16)
    block_table = torch.tensor([[0]], dtype=torch.int32)

    topk_indices, score = pytorch_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2], dtype=torch.int32),
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2,
        sparse_mode=3,
    )

    assert torch.equal(topk_indices[:, 0], torch.tensor([[0, -1], [1, 0]], dtype=torch.int32))
    assert torch.isneginf(score[0, 0, 1])
    assert torch.allclose(score[1, 0], torch.tensor([0.0, 1.0]))


def test_pytorch_quant_lightning_indexer_dequantizes_before_score():
    query = torch.tensor([[[2, 0], [0, 4]]], dtype=torch.int8)
    query_scale = torch.tensor([[0.5, 0.25]], dtype=torch.float16)
    weights = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    key = torch.tensor([[[[2, 0]], [[0, 4]]]], dtype=torch.int8)
    key_scale = torch.tensor([[[0.5], [0.25]]], dtype=torch.float16)
    block_table = torch.tensor([[0]], dtype=torch.int32)

    topk_indices, score = pytorch_quant_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2], dtype=torch.int32),
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2,
        sparse_mode=0,
    )

    assert torch.allclose(score[0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(topk_indices[0, 0], torch.tensor([1, 0], dtype=torch.int32))
