# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend import envs


def _tensor_to_int_list(tensor: torch.Tensor | None, default: list[int]) -> list[int]:
    if tensor is None:
        return default
    return [int(item) for item in tensor.detach().cpu().tolist()]


def _get_query_lens(
    query: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None,
    layout_query: str,
) -> list[int]:
    if layout_query == "BSND":
        return [query.shape[1]] * query.shape[0]

    if actual_seq_lengths_query is None:
        return [query.shape[0]]

    query_ends = _tensor_to_int_list(actual_seq_lengths_query, [])
    query_starts = [0] + query_ends[:-1]
    return [end - start for start, end in zip(query_starts, query_ends)]


def _get_key_lens(
    key: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor | None,
    layout_key: str,
    batch_size: int,
) -> list[int]:
    if layout_key == "PA_BSND":
        default_len = key.shape[0] * key.shape[1]
        return _tensor_to_int_list(actual_seq_lengths_key, [default_len] * batch_size)

    if layout_key == "BSND":
        return _tensor_to_int_list(actual_seq_lengths_key, [key.shape[1]] * batch_size)

    if actual_seq_lengths_key is None:
        return [key.shape[0]]

    key_ends = _tensor_to_int_list(actual_seq_lengths_key, [])
    key_starts = [0] + key_ends[:-1]
    return [end - start for start, end in zip(key_starts, key_ends)]


def _get_query_batch(
    query: torch.Tensor,
    weights: torch.Tensor,
    batch_idx: int,
    query_start: int,
    query_len: int,
    layout_query: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout_query == "TND":
        return query[query_start : query_start + query_len], weights[query_start : query_start + query_len]
    if layout_query == "BSND":
        return query[batch_idx, :query_len], weights[batch_idx, :query_len]
    raise NotImplementedError(f"Unsupported query layout for PyTorch lightning indexer: {layout_query}")


def _get_key_batch(
    key: torch.Tensor,
    batch_idx: int,
    key_start: int,
    key_len: int,
    layout_key: str,
    block_table: torch.Tensor | None,
) -> torch.Tensor:
    if layout_key == "PA_BSND":
        if block_table is None:
            raise ValueError("block_table is required when layout_key='PA_BSND'")
        block_size = key.shape[1]
        positions = torch.arange(key_len, device=key.device, dtype=torch.long)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        block_offsets = positions % block_size
        physical_blocks = block_table[batch_idx, logical_blocks].to(torch.long)
        return key[physical_blocks, block_offsets]

    if layout_key == "TND":
        return key[key_start : key_start + key_len]

    if layout_key == "BSND":
        return key[batch_idx, :key_len]

    raise NotImplementedError(f"Unsupported key layout for PyTorch lightning indexer: {layout_key}")


def _dump_score(
    score: torch.Tensor,
    topk_indices: torch.Tensor,
    metadata: dict[str, Any],
    score_dump_dir: str | None,
) -> None:
    dump_dir = score_dump_dir or envs.VLLM_ASCEND_LIGHTNING_INDEXER_SCORE_DUMP_DIR
    if not dump_dir:
        return

    os.makedirs(dump_dir, exist_ok=True)
    filename = f"lightning_indexer_score_{os.getpid()}_{time.time_ns()}.pt"
    path = os.path.join(dump_dir, filename)
    torch.save(
        {
            "score": score.detach().cpu(),
            "topk_indices": topk_indices.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )


def _dequantize_per_token_head(tensor: torch.Tensor, scale: torch.Tensor, name: str) -> torch.Tensor:
    if scale.shape != tensor.shape[:-1]:
        raise ValueError(
            f"{name} scale shape {tuple(scale.shape)} must match tensor shape without D {tuple(tensor.shape[:-1])}"
        )
    return tensor.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)


@torch.no_grad()
def pytorch_lightning_indexer(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    return_value: bool = False,
    score_dump_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch debug implementation of lightning_indexer.

    The first output matches the operator's sparse index output. The second
    output is the full pre-topk score tensor, padded with -inf on invalid keys.
    This is intentionally slow and should only be used for offline debugging.
    """
    if return_value:
        raise NotImplementedError("The PyTorch debug wrapper returns full score, not sparseValuesOut.")

    if layout_query not in {"TND", "BSND"}:
        raise NotImplementedError(f"Unsupported query layout: {layout_query}")
    if layout_key not in {"PA_BSND", "TND", "BSND"}:
        raise NotImplementedError(f"Unsupported key layout: {layout_key}")

    query_lens = _get_query_lens(query, actual_seq_lengths_query, layout_query)
    batch_size = len(query_lens)
    key_lens = _get_key_lens(key, actual_seq_lengths_key, layout_key, batch_size)
    if len(key_lens) != batch_size:
        raise ValueError(f"query batch size {batch_size} does not match key lens size {len(key_lens)}")

    if layout_query == "TND":
        total_query_len = query.shape[0]
    else:
        total_query_len = query.shape[0] * query.shape[1]

    n1 = query.shape[-2]
    head_dim = query.shape[-1]
    if layout_key == "PA_BSND":
        n2 = key.shape[-2]
    else:
        n2 = key.shape[-2]
    if n1 % n2 != 0:
        raise ValueError(f"query N1 ({n1}) must be divisible by key N2 ({n2})")
    group_size = n1 // n2

    max_key_len = max(key_lens) if key_lens else 0
    topk_indices = torch.full(
        (total_query_len, n2, sparse_count),
        -1,
        device=query.device,
        dtype=torch.int32,
    )
    score_out = torch.full(
        (total_query_len, n2, max_key_len),
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )

    query_start = 0
    key_start = 0
    output_start = 0
    for batch_idx, (query_len, key_len) in enumerate(zip(query_lens, key_lens)):
        query_b, weights_b = _get_query_batch(
            query,
            weights,
            batch_idx,
            query_start,
            query_len,
            layout_query,
        )
        key_b = _get_key_batch(key, batch_idx, key_start, key_len, layout_key, block_table)

        if key_b.shape[-1] != head_dim:
            raise ValueError(f"query head dim {head_dim} does not match key head dim {key_b.shape[-1]}")

        query_b = query_b.to(torch.float32).reshape(query_len, n2, group_size, head_dim)
        weights_b = weights_b.to(torch.float32).reshape(query_len, n2, group_size)
        key_b = key_b.to(torch.float32)

        for n2_idx in range(n2):
            relu_qk = F.relu(torch.einsum("qgd,kd->qgk", query_b[:, n2_idx], key_b[:, n2_idx]))
            score = torch.sum(relu_qk * weights_b[:, n2_idx, :, None], dim=1)

            for row_idx in range(query_len):
                visible_key_len = key_len
                if sparse_mode == 3:
                    visible_key_len = key_len - query_len + row_idx + 1
                visible_key_len = max(0, min(visible_key_len, key_len))

                out_row = output_start + row_idx
                if visible_key_len > 0:
                    score_out[out_row, n2_idx, :visible_key_len] = score[row_idx, :visible_key_len]

                if visible_key_len <= 0:
                    continue
                if visible_key_len < sparse_count:
                    topk_indices[out_row, n2_idx, :visible_key_len] = torch.arange(
                        visible_key_len,
                        device=query.device,
                        dtype=torch.int32,
                    )
                    continue

                row_topk = torch.topk(score[row_idx, :visible_key_len], k=sparse_count, dim=-1).indices
                topk_indices[out_row, n2_idx] = row_topk.to(torch.int32)

        query_start += query_len
        key_start += key_len if layout_key == "TND" else 0
        output_start += query_len

    if layout_query == "BSND":
        topk_indices = topk_indices.reshape(query.shape[0], query.shape[1], n2, sparse_count)
        score_out = score_out.reshape(query.shape[0], query.shape[1], n2, max_key_len)

    _dump_score(
        score_out,
        topk_indices,
        {
            "layout_query": layout_query,
            "layout_key": layout_key,
            "sparse_count": sparse_count,
            "sparse_mode": sparse_mode,
            "query_shape": tuple(query.shape),
            "key_shape": tuple(key.shape),
            "weights_shape": tuple(weights.shape),
            "query_lens": query_lens,
            "key_lens": key_lens,
        },
        score_dump_dir,
    )

    return topk_indices, score_out


@torch.no_grad()
def pytorch_quant_lightning_indexer(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor | None = None,
    actual_seq_lengths_key: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    score_dump_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch debug implementation of quant_lightning_indexer.

    Only query_quant_mode=0 and key_quant_mode=0 are supported, matching the
    current custom op contract: per-token-head scale with shape tensor.shape[:-1].
    """
    if query_quant_mode != 0:
        raise NotImplementedError(f"Unsupported query_quant_mode: {query_quant_mode}")
    if key_quant_mode != 0:
        raise NotImplementedError(f"Unsupported key_quant_mode: {key_quant_mode}")

    query_dequant = _dequantize_per_token_head(query, query_dequant_scale, "query")
    key_dequant = _dequantize_per_token_head(key, key_dequant_scale, "key")

    return pytorch_lightning_indexer(
        query=query_dequant,
        key=key_dequant,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        score_dump_dir=score_dump_dir,
    )
