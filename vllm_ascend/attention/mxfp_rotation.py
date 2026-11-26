# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed per-head signed Hadamard transforms for static MXFP attention.

Uses TurboQuantLatent's default (not legacy Lloyd-Max) seed-0 recipe.
Matrices are initialized before profiling/graph capture. For row vectors,
forward is x @ R and inverse is x @ R.T; signed R is not symmetric.
"""

import hashlib
import math

import numpy as np
import torch

HADAMARD_SEED = 0
HADAMARD_RECIPE = "turboquant_default_signed_hadamard_v1"


def signed_hadamard(head_size: int) -> torch.Tensor:
    if head_size <= 0 or head_size & (head_size - 1):
        raise ValueError("Hadamard head size must be a positive power of two")
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < head_size:
        h = np.block([[h, h], [h, -h]])
    signs = np.random.default_rng(HADAMARD_SEED).choice([-1.0, 1.0], head_size).astype(np.float32)
    return torch.from_numpy(signs[:, None] * h / math.sqrt(head_size))


def rotation_metadata(mode: str, matrix: torch.Tensor | None = None) -> dict:
    if mode not in ("none", "qk", "qkv"):
        raise ValueError("MXFP rotation must be none, qk, or qkv")
    if mode == "none":
        return {"mode": "none"}
    if matrix is None or matrix.device.type != "cpu":
        raise ValueError("Rotation provenance requires the CPU initialization matrix")
    return {
        "mode": mode,
        "recipe": HADAMARD_RECIPE,
        "seed": HADAMARD_SEED,
        "head_size": matrix.shape[0],
        "matrix_sha256": hashlib.sha256(matrix.contiguous().numpy().tobytes()).hexdigest(),
        "placement": "post_rope",
        "v_inverse": "before_output_gate" if mode == "qkv" else "none",
    }


def validate_rotation_sidecar(calibration: dict, expected: dict) -> None:
    # Old sidecars are valid only for the original unrotated path. Even QK-only
    # experiments use a new calibration, with explicit distribution provenance.
    actual = calibration.get("rotation", {"mode": "none"})
    if actual != expected:
        raise ValueError(f"Static V sidecar rotation mismatch: {actual} != {expected}")


def rotate_heads(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """FP32 matmul and cast, matching TurboQuant; supports flattened head axes."""
    return (tensor.float().reshape(-1, matrix.shape[0]) @ matrix).to(tensor.dtype).reshape(tensor.shape)
