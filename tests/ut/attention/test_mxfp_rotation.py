# SPDX-License-Identifier: Apache-2.0
"""Mathematical and observer contracts for the optional Hadamard path."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.mxfp_rotation import (
    rotate_heads, rotation_metadata, signed_hadamard, validate_rotation_sidecar,
)
from vllm_ascend.attention.mxfp_static import AscendMXFPCalibrationImpl, calibration_path


@pytest.mark.parametrize("dim", [128, 256])
def test_turboquant_recipe_orthogonality_and_inverse(dim):
    r = signed_hadamard(dim)
    # Explicit Kronecker construction independently checks the TurboQuant recipe.
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(dim.bit_length() - 1):
        h = np.kron(h, np.array([[1, 1], [1, -1]], dtype=np.float32))
    signs = np.random.default_rng(0).choice([-1.0, 1.0], dim).astype(np.float32)
    torch.testing.assert_close(r, torch.from_numpy(signs[:, None] * h / np.sqrt(dim)), rtol=0, atol=0,
                               check_dtype=False)
    torch.testing.assert_close(r @ r.T, torch.eye(dim), atol=3e-7, rtol=0)
    assert not torch.equal(r, r.T)
    x = torch.randn(3, 2 * dim, generator=torch.Generator().manual_seed(41))
    torch.testing.assert_close(rotate_heads(rotate_heads(x, r), r.T), x, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(rotate_heads(rotate_heads(x, r), r), x)


@pytest.mark.parametrize("dim", [0, -1, 3, 255])
def test_reject_non_power_of_two(dim):
    with pytest.raises(ValueError, match="power of two"):
        signed_hadamard(dim)


@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_attention_and_gate_equivalence(mode):
    g = torch.Generator().manual_seed(91)
    q, k, v = [torch.randn(s, 256, generator=g) for s in (4, 19, 19)]
    gate = torch.sigmoid(torch.randn(4, 256, generator=g))
    r = signed_hadamard(256)
    reference = (q @ k.T / 16).softmax(-1) @ v
    rotated_v = rotate_heads(v, r) if mode == "qkv" else v
    result = (rotate_heads(q, r) @ rotate_heads(k, r).T / 16).softmax(-1) @ rotated_v
    if mode == "qkv":
        result = rotate_heads(result, r.T)
    torch.testing.assert_close(result * gate, reference * gate, atol=2e-6, rtol=2e-5)


def test_sidecar_rotation_provenance():
    qk = rotation_metadata("qk", signed_hadamard(256))
    qkv = rotation_metadata("qkv", signed_hadamard(256))
    validate_rotation_sidecar({}, {"mode": "none"})
    validate_rotation_sidecar({"rotation": qkv}, qkv)
    for old, wanted in [({}, qk), ({"rotation": qk}, qkv), ({"rotation": qkv}, {"mode": "none"}),
                        ({"rotation": {**qkv, "matrix_sha256": "wrong"}}, qkv)]:
        with pytest.raises(ValueError, match="rotation mismatch"):
            validate_rotation_sidecar(old, wanted)


@pytest.mark.parametrize("mode", ["qk", "qkv"])
def test_calibration_observes_correct_v_basis_and_restores_output(tmp_path, mode):
    impl = object.__new__(AscendMXFPCalibrationImpl)
    impl._mxfp_static_config = {"scale_dir": str(tmp_path), "rotation": mode}
    impl.num_kv_heads, impl.head_size = 1, 256
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="checkpoint"))
    r = signed_hadamard(256)
    impl._mxfp_rotation_metadata = rotation_metadata(mode, r)
    layer = SimpleNamespace(layer_name="model.layers.3.self_attn.attn", _mxfp_rotation=r,
                            _mxfp_inverse_rotation=r.T.contiguous())
    gen = torch.Generator().manual_seed(17)
    q, k, v = [torch.randn(3, 1, 256, generator=gen) for _ in range(3)]
    v[2] = 1e6  # Padding must never enter the observer.
    expected_v = rotate_heads(v, r) if mode == "qkv" else v
    out = torch.full_like(v, -123)
    base_out = torch.randn(2, 1, 256, generator=gen)
    def fake_forward(_, query, key, value, cache, metadata, **kwargs):
        torch.testing.assert_close(query, rotate_heads(q, r))
        torch.testing.assert_close(key, rotate_heads(k, r))
        torch.testing.assert_close(value, expected_v)
        out[:2] = rotate_heads(base_out, r) if mode == "qkv" else base_out
        return out
    (tmp_path / "calibration.enabled").touch()
    with patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_rank", return_value=0), \
         patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_world_size", return_value=2), \
         patch("vllm_ascend.attention.mxfp_static.AscendAttentionBackendImpl.forward", side_effect=fake_forward):
        result = impl.forward(layer, q, k, v, None, SimpleNamespace(num_actual_tokens=2), output=out)
    assert result is out and torch.equal(out[2], torch.full_like(out[2], -123))
    torch.testing.assert_close(out[:2], base_out, atol=2e-6, rtol=2e-6)
    data = torch.load(calibration_path(str(tmp_path), layer.layer_name, 0), weights_only=True)
    torch.testing.assert_close(data["v_max"], expected_v[:2].abs().amax(0))
    assert data["rotation"] == impl._mxfp_rotation_metadata and data["observed_tokens"] == 2
