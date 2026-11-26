# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.mxfp_rotation import signed_hadamard
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.methods.kv_cache.mxfp_c4 import (
    AscendC4MXFPKVCacheAttentionMethod,
    _c4_rotation_loader,
    _c4_scale_loader,
    validate_c4_config,
)

PREFIX = "model.layers.3.self_attn"
CONFIG = {
    "version": 1, "mode": "qkv", "head_dim": 256, "seed": 0,
    "recipe": "turboquant_default_signed_hadamard_v1",
    "matrix_sha256": "ca70b9def213b38eb4066626b2808ec33d5694c38f7f92085703936e8925d22d",
    "placement": "post_rope_before_cache", "inverse_placement": "before_output_gate",
    "compute_dtype": "float32", "output_dtype": "bfloat16",
    "q": {"dtype": "mxfp4", "block_size": 32, "scale_alg": 2, "dst_type_max": 7.25},
    "k": {"dtype": "mxfp4", "block_size": 32, "scale_alg": 2, "dst_type_max": 7.25},
    "v": {"dtype": "mxfp4", "scope": "per_head_channel", "calibration": "static",
          "exponent": "floor(log2(amax*8/7.25+9.6e-7))-2",
          "statistics_dtype": "float32", "round_mode": "rint"},
    "quant_schema": 2, "calibration_attention": "bf16_eager_after_qkv_quant_dequant",
    "p_quantization_simulated": False, "propagation": "recompute_after_qkv_quantization",
    "scope": "target_full_attention_only",
}


def description():
    return {PREFIX + ".quant_type": "QK_MXFP4_DYNAMIC_V_MXFP4_PER_CHANNEL", PREFIX + ".c4_config": deepcopy(CONFIG)}


def runtime(dtype="mxfp4"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=dtype), speculative_config=None,
        kv_transfer_config=None, model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def layer_and_method():
    from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl

    layer = torch.nn.Module()
    layer.num_kv_heads = 2
    layer.head_size = layer.head_size_v = 256
    layer.impl = object.__new__(AscendC8MXFPAttentionBackendImpl.__base__)
    method = AscendC4MXFPKVCacheAttentionMethod(description(), PREFIX + ".attn")
    with patch("vllm_ascend.quantization.methods.kv_cache.mxfp_c4.get_current_vllm_config", return_value=runtime()):
        method.create_weights(layer)
    return layer, method


@pytest.mark.parametrize("field,value", [("mode", "qk"), ("quant_schema", 0), ("inverse_placement", "after_gate"),
                                        ("p_quantization_simulated", True), ("head_dim", 128)])
def test_rejects_incompatible_contract(field, value):
    config = deepcopy(CONFIG)
    config[field] = value
    with pytest.raises(ValueError, match="Unsupported"):
        validate_c4_config(config)


@pytest.mark.parametrize("field", ["q", "k"])
def test_rejects_old_alg0_or_c7(field):
    config = deepcopy(CONFIG)
    config[field]["scale_alg"] = 0
    with pytest.raises(ValueError):
        validate_c4_config(config)
    config = deepcopy(CONFIG)
    config[field]["dst_type_max"] = 7.0
    with pytest.raises(ValueError):
        validate_c4_config(config)


def test_checkpoint_parameters_loaded_and_inverse_prepared():
    layer, method = layer_and_method()
    scale = torch.arange(512, dtype=torch.int32).remainder(9).add(119).to(torch.uint8).view(-1, 1)
    matrix = signed_hadamard(256)
    layer.v_cache_scale.weight_loader(layer.v_cache_scale, scale)
    layer._mxfp_rotation.weight_loader(layer._mxfp_rotation, matrix)
    with patch("vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_current_vllm_config", return_value=runtime()):
        method.process_weights_after_loading(layer)
    assert torch.equal(layer.v_cache_scale, scale.flatten())
    assert torch.equal(layer.v_cache_scale_float_reciprocal, torch.exp2(127 - scale.flatten().float()).bfloat16())
    assert torch.equal(layer._mxfp_rotation, matrix)
    assert torch.equal(layer._mxfp_inverse_rotation, matrix.T)
    assert layer.impl.q_quant_options == {"block_size": 32, "scale_alg": 2, "dst_type_max": 7.25}
    assert layer.impl.k_quant_max == 7.25
    assert layer.impl.qfa_quant_schema == 2
    assert layer._mxfp_checkpoint_loaded


def test_missing_scale_and_missing_or_corrupt_matrix_fail():
    layer, method = layer_and_method()
    with pytest.raises(ValueError, match="V scale"):
        method.process_weights_after_loading(layer)
    layer.v_cache_scale.weight_loader(layer.v_cache_scale, torch.full((512, 1), 121, dtype=torch.uint8))
    with pytest.raises(ValueError, match="rotation"):
        method.process_weights_after_loading(layer)
    matrix = signed_hadamard(256)
    matrix[0, 0] *= -1
    layer._mxfp_rotation.weight_loader(layer._mxfp_rotation, matrix)
    with pytest.raises(ValueError, match="rotation"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("bad", [0, 255])
def test_invalid_scale_is_not_sanitized(bad):
    with pytest.raises(ValueError):
        _c4_scale_loader(torch.empty(512, dtype=torch.uint8), torch.full((512, 1), bad, dtype=torch.uint8))


def test_rotation_loader_rejects_cast_or_transposed_shape():
    with pytest.raises(ValueError):
        _c4_rotation_loader(torch.empty(256, 256), signed_hadamard(256).bfloat16())
    with pytest.raises(ValueError):
        _c4_rotation_loader(torch.empty(256, 256), torch.empty(128, 512))


def test_checkpoint_mapping_and_dtype_guard():
    config = AscendModelSlimConfig(description())
    name = "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config_or_none"
    with patch(name, return_value=runtime()):
        mapper = config.get_cache_scale_mapper()
        mapped = dict(mapper.apply([(PREFIX + ".fa_v.scale", 1), (PREFIX + ".c4_rotation.matrix", 2),
                                    (PREFIX + ".fa_v.offset", 3)]))
        assert mapped == {PREFIX + ".attn.v_cache_scale": 1, PREFIX + ".attn._mxfp_rotation": 2}
    for dtype in ["auto", "bfloat16", "mxfp8"]:
        # Mapping is independent of the current model context, including the
        # BF16 MTP draft. The actual target method validates cache dtype.
        with patch(name, return_value=runtime(dtype)):
            assert config.get_cache_scale_mapper() is not None
        layer, method = layer_and_method()
        with patch("vllm_ascend.quantization.methods.kv_cache.mxfp_c4.get_current_vllm_config",
                   return_value=runtime(dtype)), pytest.raises(ValueError, match="mxfp4"):
            method.create_weights(layer)

@pytest.mark.parametrize("dtype", ["mxfp4", "bfloat16"])
def test_c4_fakquant_header_never_uses_mla_mapping(dtype):
    # ModelSlim AscendV1 writes this legacy header even for the C4 recipe.
    # Test all six FA tensors, not just the directly consumed V scale.
    desc = description()
    desc["fa_quant_type"] = "FAKQuant"
    weights = {}
    for head in ["q", "k", "v"]:
        for field in ["scale", "offset"]:
            key = PREFIX + ".fa_" + head + "." + field
            desc[key] = "FLOAT"
            weights[key] = torch.ones(512, 1)
    weights[PREFIX + ".c4_rotation.matrix"] = signed_hadamard(256)
    config = AscendModelSlimConfig(desc)
    name = "vllm_ascend.quantization.configs.modelslim_config.get_current_vllm_config_or_none"
    with patch(name, return_value=runtime(dtype)):
        mapper = config.get_cache_scale_mapper()
        mapped = dict(mapper.apply(weights.items()))
        assert set(mapped) == {PREFIX + ".attn.v_cache_scale", PREFIX + ".attn._mxfp_rotation"}
        assert not config.enable_fa_quant
        # The nested Qwen loader can apply the mapper again.
        assert set(dict(mapper.apply(mapped.items()))) == set(mapped)