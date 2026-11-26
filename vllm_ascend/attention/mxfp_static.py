# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-channel V calibration and static MXFP KV for unquantized checkpoints.

Calibration is eager-only and explicitly armed after startup with a marker
file, so profiling/warmup tensors never contaminate the observations.
"""

from __future__ import annotations

import importlib
import math
import re
from pathlib import Path

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.attention.mxfp_rotation import (
    rotate_heads,
    rotation_metadata,
    signed_hadamard,
    validate_rotation_sidecar,
)

logger = init_logger(__name__)


def calibration_path(directory: str, layer_name: str, rank: int) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", layer_name)
    return Path(directory) / f"{safe_name}.rank{rank}.pt"


def encode_static_v_scale(channel_max: torch.Tensor, divisor: float) -> torch.Tensor:
    """Finite power-of-two scale, with neutral encoding for zero channels."""
    values = channel_max.float()
    if divisor <= 0 or not torch.isfinite(torch.tensor(divisor)):
        raise ValueError("The static V scale divisor must be finite and positive")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("V calibration maxima must be finite and nonnegative")
    exponent = torch.ceil(torch.log2(values.clamp_min(torch.finfo(torch.float32).tiny) / divisor))
    exponent = torch.where(values == 0, torch.zeros_like(exponent), exponent)
    if (exponent > 127).any():
        raise ValueError("V calibration exceeds finite E8M0 range")
    return (exponent.clamp(min=-126, max=127) + 127).to(torch.uint8)


def load_static_v_scale(calibration, dtype, shape, divisor_override=None):
    """Load an explicitly encoded scale, or preserve the legacy amax rule."""
    maximum = calibration["v_max"]
    if tuple(maximum.shape) != tuple(shape):
        raise ValueError("Static V sidecar shape mismatch")
    if not torch.isfinite(maximum).all() or (maximum < 0).any():
        raise ValueError("V calibration maxima must be finite and nonnegative")
    if "v_scale_e8m0" not in calibration:
        divisor = 6.0 if dtype == "mxfp4" else 448.0
        return encode_static_v_scale(maximum, divisor if divisor_override is None else float(divisor_override))
    if divisor_override is not None:
        raise ValueError("v_scale_divisor cannot override explicitly encoded V scales")
    if calibration.get("quant_dtype") != dtype:
        raise ValueError("Static V sidecar quant_dtype mismatch")
    scale = calibration["v_scale_e8m0"]
    if scale.dtype != torch.uint8 or tuple(scale.shape) != tuple(shape):
        raise ValueError("Encoded V scale must be uint8 with the local KV-head/channel shape")
    # 255 is NaN; 0 needs an inverse beyond finite FP32 in this backend.
    if ((scale == 0) | (scale == 255)).any():
        raise ValueError("Encoded V scale must have a finite FP32 scale and inverse")
    return scale


def qfa_ops(vllm_config):
    config = (vllm_config.additional_config or {}).get("mxfp_static", {})
    package = config.get("qfa_package", "cann_ops_transformer")
    # Each package owns its matching C++ bridge and registers the common ops.
    return importlib.import_module(f"{package}.ops")


class AscendMXFPCalibrationImpl(AscendAttentionBackendImpl):
    """Ordinary BF16 attention plus an explicitly armed V observer."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, **kwargs):
        config = self._mxfp_static_config
        rotation = config.get("rotation", "none")
        if rotation != "none" and attn_metadata is not None:
            query = rotate_heads(query, layer._mxfp_rotation)
            if key is not None:
                key = rotate_heads(key, layer._mxfp_rotation)
            if rotation == "qkv" and value is not None:
                value = rotate_heads(value, layer._mxfp_rotation)
        marker = Path(config["scale_dir"]) / "calibration.enabled"
        if attn_metadata is not None and value is not None and marker.is_file():
            count = attn_metadata.num_actual_tokens
            if count:
                observed = value[:count].view(count, self.num_kv_heads, self.head_size).float().abs().amax(dim=0)
                previous = getattr(self, "_mxfp_v_max", None)
                self._mxfp_v_max = observed if previous is None else torch.maximum(previous, observed)
                self._mxfp_observed_tokens = getattr(self, "_mxfp_observed_tokens", 0) + count
                rank = get_tensor_model_parallel_rank()
                destination = calibration_path(config["scale_dir"], layer.layer_name, rank)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                torch.save(
                    {
                        "v_max": self._mxfp_v_max.cpu(),
                        "layer_name": layer.layer_name,
                        "rank": rank,
                        "tp_size": get_tensor_model_parallel_world_size(),
                        "model": str(self.vllm_config.model_config.model),
                        "observed_tokens": self._mxfp_observed_tokens,
                        "rotation": getattr(self, "_mxfp_rotation_metadata", {"mode": "none"}),
                    },
                    temporary,
                )
                temporary.replace(destination)
        output = super().forward(layer, query, key, value, kv_cache, attn_metadata, **kwargs)
        if rotation == "qkv" and attn_metadata is not None:
            count = attn_metadata.num_actual_tokens
            output[:count].copy_(rotate_heads(output[:count], layer._mxfp_inverse_rotation))
        return output


def _validate_mxfp4_mtp(speculative) -> None:
    if speculative is None:
        return
    # Upstream normalizes the legacy qwen3_5_mtp method to mtp. The draft
    # model type preserves the architecture that determines cache support.
    draft_config = getattr(speculative, "draft_model_config", None)
    draft_hf_config = getattr(draft_config, "hf_config", None)
    if speculative.method not in ("mtp", "qwen3_5_mtp") or (
        getattr(draft_hf_config, "model_type", None) != "qwen3_5_mtp"
    ):
        raise NotImplementedError("Static MXFP4 currently supports Qwen3.5/3.6 native MTP only")
    if speculative.kv_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Static MXFP4 MTP requires speculative_config.kv_cache_dtype=auto or bfloat16")


def install_mxfp_static_layers(vllm_config) -> None:
    """Install before MRV2 profiles memory and constructs cache specs."""
    config = (vllm_config.additional_config or {}).get("mxfp_static", {})
    if config.get("mode") == "checkpoint":
        if set(config) - {"mode", "qfa_package"}:
            raise ValueError("C4 checkpoint parameters come from model metadata and cannot be overridden")
        if vllm_config.cache_config.cache_dtype != "mxfp4":
            raise ValueError("C4 checkpoint mode requires --kv-cache-dtype mxfp4")
        context = vllm_config.compilation_config.static_forward_context
        layers = [
            layer for layer in context.values()
            if isinstance(layer, Attention) and not layer.is_draft_layer
            and getattr(layer, "attn_type", "decoder") == "decoder"
        ]
        if not layers or not all(getattr(layer, "_mxfp_checkpoint_loaded", False) for layer in layers):
            raise ValueError("Every target GQA layer must load its C4 checkpoint scale and rotation")
        logger.info("MXFP C4 checkpoint verified on %d target GQA layers", len(layers))
        return
    if not config:
        if vllm_config.cache_config.cache_dtype == "mxfp4":
            raise ValueError("MXFP4 requires mxfp_static.scale_dir with calibrated V maxima")
        return
    allowed = {"mode", "scale_dir", "qfa_package", "v_scale_divisor", "quant_schema", "k_quant_max", "rotation"}
    if set(config) - allowed:
        raise ValueError(f"Unknown mxfp_static options: {sorted(set(config) - allowed)}")
    mode = config.get("mode", "static")
    if mode not in ("calibrate", "static") or not config.get("scale_dir"):
        raise ValueError("mxfp_static requires mode=calibrate/static and scale_dir")
    dtype = vllm_config.cache_config.cache_dtype
    if mode == "calibrate":
        if not vllm_config.model_config.enforce_eager or dtype not in ("auto", "bfloat16"):
            raise ValueError("MXFP V calibration requires eager execution and BF16 KV cache")
    elif dtype not in ("mxfp8", "mxfp4"):
        raise ValueError("Static MXFP setup requires --kv-cache-dtype mxfp8 or mxfp4")
    rotation = config.get("rotation", "none")
    if rotation not in ("none", "qk", "qkv"):
        raise ValueError("MXFP rotation must be none, qk, or qkv")
    if rotation != "none" and mode != "calibrate" and dtype != "mxfp4":
        raise ValueError("Hadamard rotation requires BF16 calibration or static MXFP4")
    if "k_quant_max" in config and dtype != "mxfp4":
        raise ValueError("k_quant_max requires static MXFP4 KV cache")
    if dtype == "mxfp4":
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("MXFP4 currently requires local execution without KV transfer")
        _validate_mxfp4_mtp(vllm_config.speculative_config)
        k_quant_max = config.get("k_quant_max", 7.0)
        if isinstance(k_quant_max, bool) or not isinstance(k_quant_max, (int, float)):
            raise ValueError("MXFP4 k_quant_max must be a finite number in [6, 12]")
        if not math.isfinite(k_quant_max) or not 6.0 <= k_quant_max <= 12.0:
            raise ValueError("MXFP4 k_quant_max must be a finite number in [6, 12]")
        if config.get("quant_schema", 2) not in (0, 1, 2):
            raise ValueError("MXFP4 quant_schema must be 0, 1, or 2")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("Static MXFP sidecar loading currently requires PP=1")
    rank = get_tensor_model_parallel_rank()
    tp = get_tensor_model_parallel_world_size()
    context = vllm_config.compilation_config.static_forward_context
    count = 0
    for name, layer in context.items():
        if not isinstance(layer, Attention) or not isinstance(layer.impl, AscendAttentionBackendImpl):
            continue
        if getattr(layer, "attn_type", "decoder") != "decoder" or layer.is_draft_layer:
            # The sidecars describe target-model V only. MTP keeps its native
            # BF16 backend/cache; its distribution is not target-layer data.
            continue
        matrix = None
        if rotation != "none":
            if layer.head_size != layer.head_size_v:
                raise ValueError("Hadamard MXFP requires equal QK/V head dimensions")
            matrix = signed_hadamard(layer.head_size)
            device = f"npu:{torch.npu.current_device()}"
            layer.register_buffer("_mxfp_rotation", matrix.to(device), persistent=False)
            layer.register_buffer("_mxfp_inverse_rotation", matrix.T.contiguous().to(device), persistent=False)
        expected_rotation = rotation_metadata(rotation, matrix)
        if mode == "calibrate":
            layer.impl.__class__ = AscendMXFPCalibrationImpl
            layer.impl._mxfp_static_config = config
        else:
            from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import AscendC8MXFPKVCacheAttentionMethod

            source = calibration_path(config["scale_dir"], name, rank)
            calibration = torch.load(source, map_location="cpu", weights_only=True)
            if calibration.get("layer_name") != name or calibration.get("rank") != rank:
                raise ValueError(f"Static V sidecar layer/rank mismatch: {source}")
            if calibration.get("tp_size") != tp or calibration.get("model") != str(vllm_config.model_config.model):
                raise ValueError(f"Static V sidecar model/TP mismatch: {source}")
            validate_rotation_sidecar(calibration, expected_rotation)
            scale = load_static_v_scale(
                calibration, dtype, (layer.num_kv_heads, layer.head_size_v), config.get("v_scale_divisor")
            )
            if dtype == "mxfp4":
                from vllm_ascend.quantization.methods.kv_cache.mxfp_c4 import AscendC4MXFPKVCacheAttentionMethod

                method = AscendC4MXFPKVCacheAttentionMethod({}, name)
                if hasattr(layer, "v_cache_scale"):
                    raise NotImplementedError("MXFP4 sidecars currently require an unquantized checkpoint")
            else:
                method = AscendC8MXFPKVCacheAttentionMethod({}, name)
            if not hasattr(layer, "v_cache_scale"):
                method.create_weights(layer)
            else:
                from vllm_ascend.attention.attention_c8_mxfp import (
                    AscendC8MXFPAttentionBackend,
                    AscendC8MXFPAttentionBackendImpl,
                )

                layer.attn_backend = AscendC8MXFPAttentionBackend
                layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
                layer.impl.enable_hamming_sparse = False
            layer.v_cache_scale.data = scale.flatten().to(device=f"npu:{torch.npu.current_device()}")
            method.process_weights_after_loading(layer)
            if dtype == "mxfp4":
                layer.impl.k_quant_max = float(k_quant_max)
        layer.impl._mxfp_rotation_mode = rotation
        layer.impl._mxfp_rotation_metadata = expected_rotation
        count += 1
    if not count:
        raise ValueError("mxfp_static did not find any compatible GQA attention layers")
    logger.info("MXFP static V mode=%s installed on %d GQA layers, TP rank %d", mode, count, rank)
