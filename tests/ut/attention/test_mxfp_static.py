# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.mxfp_static import (
    AscendMXFPCalibrationImpl,
    calibration_path,
    encode_static_v_scale,
    load_static_v_scale,
)


@pytest.mark.parametrize("divisor", [6.0, 448.0])
def test_static_scale_covers_maximum_without_clipping(divisor):
    maxima = torch.tensor([0, divisor / 4, divisor, divisor * 1.01, divisor * 2])
    encoded = encode_static_v_scale(maxima, divisor)
    assert encoded.tolist() == [127, 125, 127, 128, 128]
    scale = torch.exp2(encoded.float() - 127)
    assert torch.all(maxima / scale <= divisor)
    # The next smaller power of two would clip every nonzero channel.
    assert torch.all(maxima[1:] / (scale[1:] / 2) > divisor)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_static_scale_rejects_invalid_observations(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        encode_static_v_scale(torch.tensor([bad]), 6.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_static_scale_rejects_invalid_divisor(bad):
    with pytest.raises(ValueError, match="divisor"):
        encode_static_v_scale(torch.ones(1), bad)


def test_calibration_arming_padding_and_maximum_accumulation(tmp_path):
    impl = object.__new__(AscendMXFPCalibrationImpl)
    impl._mxfp_static_config = {"scale_dir": str(tmp_path)}
    impl.num_kv_heads, impl.head_size = 1, 2
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="checkpoint"))
    layer = SimpleNamespace(layer_name="model.layers.3.self_attn.attn")
    metadata = SimpleNamespace(num_actual_tokens=2)
    value = torch.tensor([[[1.0, -4.0]], [[-3.0, 2.0]], [[999.0, 999.0]]])
    destination = calibration_path(str(tmp_path), layer.layer_name, 1)
    with (
        patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_rank", return_value=1),
        patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm_ascend.attention.mxfp_static.AscendAttentionBackendImpl.forward", return_value="output"),
    ):
        assert impl.forward(layer, None, None, value, None, metadata) == "output"
        assert not destination.exists()
        (tmp_path / "calibration.enabled").touch()
        # Profiling has no metadata and must still be excluded after arming.
        impl.forward(layer, None, None, value, None, None)
        assert not destination.exists()
        impl.forward(layer, None, None, value, None, metadata)
        impl.forward(layer, None, None, torch.tensor([[[5.0, 1.0]]]), None, SimpleNamespace(num_actual_tokens=1))
    saved = torch.load(destination, weights_only=True)
    assert saved["v_max"].tolist() == [[5.0, 4.0]]
    assert saved["observed_tokens"] == 3
    assert saved["tp_size"] == 2 and saved["rank"] == 1
    assert saved["model"] == "checkpoint" and saved["layer_name"] == layer.layer_name
    assert not destination.with_suffix(".tmp").exists()


@pytest.mark.parametrize("target", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("draft", ["auto", "bfloat16", None])
def test_draft_cache_quantization_is_independent(target, draft):
    from vllm_ascend.utils import is_mxfp_kv_quant

    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=target),
        speculative_config=SimpleNamespace(kv_cache_dtype=draft),
    )
    assert is_mxfp_kv_quant(config, SimpleNamespace(is_draft=False))
    assert is_mxfp_kv_quant(config, SimpleNamespace(is_draft=True)) == (draft is None)


def test_calibration_does_not_install_on_draft_layers():
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.attention.mxfp_static import install_mxfp_static_layers

    class Layer:
        def __init__(self, draft):
            self.is_draft_layer = draft
            self.impl = object.__new__(AscendAttentionBackendImpl)

    target, draft = Layer(False), Layer(True)
    config = SimpleNamespace(
        additional_config={"mxfp_static": {"mode": "calibrate", "scale_dir": "unused"}},
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={"target": target, "draft": draft}),
    )
    with (
        patch("vllm_ascend.attention.mxfp_static.Attention", Layer),
        patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm_ascend.attention.mxfp_static.get_tensor_model_parallel_world_size", return_value=2),
    ):
        install_mxfp_static_layers(config)
    assert type(target.impl) is AscendMXFPCalibrationImpl
    assert type(draft.impl) is AscendAttentionBackendImpl


@pytest.mark.parametrize("method", ["mtp", "qwen3_5_mtp"])
@pytest.mark.parametrize("dtype", ["auto", "bfloat16"])
def test_native_mtp_accepts_normalized_method(method, dtype):
    from vllm_ascend.attention.mxfp_static import _validate_mxfp4_mtp

    config = SimpleNamespace(
        method=method,
        kv_cache_dtype=dtype,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_mtp")),
    )
    _validate_mxfp4_mtp(config)


@pytest.mark.parametrize(
    "method,model_type,dtype,error",
    [
        ("mtp", "deepseek_mtp", "auto", NotImplementedError),
        ("draft_model", "qwen3_5_mtp", "auto", NotImplementedError),
        ("mtp", "qwen3_5_mtp", "mxfp4", ValueError),
        ("mtp", "qwen3_5_mtp", None, ValueError),
    ],
)
def test_native_mtp_rejects_unsupported_draft(method, model_type, dtype, error):
    from vllm_ascend.attention.mxfp_static import _validate_mxfp4_mtp

    config = SimpleNamespace(
        method=method,
        kv_cache_dtype=dtype,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )
    with pytest.raises(error):
        _validate_mxfp4_mtp(config)


@pytest.mark.parametrize("dtype,divisor", [("mxfp8", 448.0), ("mxfp4", 6.0)])
def test_explicit_scale_is_not_recomputed_and_legacy_is_unchanged(dtype, divisor):
    maximum = torch.tensor([[3.0, 7.0]])
    legacy = {"v_max": maximum}
    assert torch.equal(load_static_v_scale(legacy, dtype, (1, 2)), encode_static_v_scale(maximum, divisor))
    encoded = torch.tensor([[120, 121]], dtype=torch.uint8)
    explicit = {**legacy, "quant_dtype": dtype, "v_scale_e8m0": encoded}
    assert torch.equal(load_static_v_scale(explicit, dtype, (1, 2)), encoded)
    with pytest.raises(ValueError, match="cannot override"):
        load_static_v_scale(explicit, dtype, (1, 2), divisor)
    with pytest.raises(ValueError, match="quant_dtype"):
        load_static_v_scale(explicit, "mxfp4" if dtype == "mxfp8" else "mxfp8", (1, 2))


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[0, 127]], dtype=torch.uint8),
        torch.tensor([[255, 127]], dtype=torch.uint8),
        torch.tensor([[127.0, 127.0]]),
        torch.tensor([127, 127], dtype=torch.uint8),
    ],
)
def test_explicit_scale_rejects_invalid_encoding(bad):
    data = {"v_max": torch.ones(1, 2), "quant_dtype": "mxfp8", "v_scale_e8m0": bad}
    with pytest.raises(ValueError, match="Encoded V scale"):
        load_static_v_scale(data, "mxfp8", (1, 2))


@pytest.mark.parametrize("maximum", [True, "7.25", None, 0.0, 5.99, 12.01, float("nan"), float("inf")])
def test_c4_k_quant_max_rejects_bad_options_before_loading(maximum):
    from vllm_ascend.attention.mxfp_static import install_mxfp_static_layers
    config = SimpleNamespace(
        additional_config={"mxfp_static": {"mode": "static", "scale_dir": "unused", "k_quant_max": maximum}},
        cache_config=SimpleNamespace(cache_dtype="mxfp4"),
        kv_transfer_config=None, speculative_config=None,
    )
    with pytest.raises(ValueError, match="k_quant_max"):
        install_mxfp_static_layers(config)


def test_c8_rejects_c4_k_quant_option():
    from vllm_ascend.attention.mxfp_static import install_mxfp_static_layers
    config = SimpleNamespace(
        additional_config={"mxfp_static": {"mode": "static", "scale_dir": "unused", "k_quant_max": 7.25}},
        cache_config=SimpleNamespace(cache_dtype="mxfp8"),
    )
    with pytest.raises(ValueError, match="k_quant_max requires"):
        install_mxfp_static_layers(config)