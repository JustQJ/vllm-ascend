# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed C4 cache, with optional ModelSlim QKV Hadamard checkpoint weights."""

import hashlib
from copy import deepcopy

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from .mxfp_c8 import AscendC8MXFPKVCacheAttentionMethod, _quant_weight_loader

logger = init_logger(__name__)
C4_QUANT_TYPE = "QK_MXFP4_DYNAMIC_V_MXFP4_PER_CHANNEL"


def validate_c4_config(config):
    """Accept the exported version-1 contract without silently changing it."""
    from vllm_ascend.attention.mxfp_rotation import HADAMARD_RECIPE, signed_hadamard

    expected = {
        "version": 1, "mode": "qkv", "head_dim": 256, "seed": 0,
        "recipe": HADAMARD_RECIPE, "placement": "post_rope_before_cache",
        "inverse_placement": "before_output_gate", "compute_dtype": "float32",
        "output_dtype": "bfloat16", "quant_schema": 2,
        "calibration_attention": "bf16_eager_after_qkv_quant_dequant",
        "p_quantization_simulated": False,
        "propagation": "recompute_after_qkv_quantization",
        "scope": "target_full_attention_only",
        "q": {"dtype": "mxfp4", "block_size": 32, "scale_alg": 2, "dst_type_max": 7.25},
        "k": {"dtype": "mxfp4", "block_size": 32, "scale_alg": 2, "dst_type_max": 7.25},
        "v": {"dtype": "mxfp4", "scope": "per_head_channel", "calibration": "static",
              "exponent": "floor(log2(amax*8/7.25+9.6e-7))-2",
              "statistics_dtype": "float32", "round_mode": "rint"},
    }
    matrix = signed_hadamard(expected["head_dim"])
    expected["matrix_sha256"] = hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
    if config != expected:
        raise ValueError("Unsupported ModelSlim C4 configuration; Q/K C7.25, QKV Hadamard and schema2 are required")
    return deepcopy(config)


def _c4_scale_loader(param, loaded_weight):
    if loaded_weight.dtype != torch.uint8 or loaded_weight.numel() % 256:
        raise ValueError("C4 V scale must contain uint8 E8M0 bytes for complete 256-channel heads")
    values = loaded_weight.detach().cpu()
    if ((values == 0) | (values == 255)).any():
        raise ValueError("C4 V scales must be finite with finite reciprocals")
    _quant_weight_loader(param, loaded_weight)


def _c4_rotation_loader(param, loaded_weight):
    if loaded_weight.dtype != torch.float32 or tuple(loaded_weight.shape) != tuple(param.shape):
        raise ValueError("C4 rotation must be an FP32 [head_dim, head_dim] matrix")
    param.data.copy_(loaded_weight)


class AscendC4MXFPKVCacheAttentionMethod(AscendC8MXFPKVCacheAttentionMethod):
    def __init__(self, quant_description, prefix):
        super().__init__(quant_description, prefix)
        parent = prefix.removesuffix(".attn")
        config = quant_description.get(parent + ".c4_config")
        self.c4_config = None if config is None else validate_c4_config(config)

    def create_weights(self, layer):
        from vllm_ascend.attention.attention_c4_mxfp import (
            AscendC4MXFPAttentionBackend,
            AscendC4MXFPAttentionBackendImpl,
        )

        super().create_weights(layer)
        layer.kv_cache_torch_dtype = torch.uint8
        layer.attn_backend = AscendC4MXFPAttentionBackend
        layer.impl.__class__ = AscendC4MXFPAttentionBackendImpl
        if self.c4_config is None:
            return  # Existing unquantized-checkpoint sidecar path.
        config = get_current_vllm_config()
        from vllm_ascend.attention.mxfp_static import _validate_mxfp4_mtp

        _validate_mxfp4_mtp(config.speculative_config)
        if config.cache_config.cache_dtype != "mxfp4" or config.kv_transfer_config is not None:
            raise ValueError("ModelSlim C4 requires --kv-cache-dtype mxfp4 and local KV cache")
        if layer.head_size != 256 or layer.head_size_v != 256 or config.model_config.dtype != torch.bfloat16:
            raise ValueError("ModelSlim C4 version 1 requires BF16 attention with D=Dv=256")
        layer.v_cache_scale.data.fill_(255)  # Missing checkpoint scales must fail, never default to 1.
        layer.v_cache_scale.weight_loader = _c4_scale_loader
        matrix = torch.nn.Parameter(torch.full((256, 256), float("nan"), dtype=torch.float32), requires_grad=False)
        matrix.weight_loader = _c4_rotation_loader
        layer.register_parameter("_mxfp_rotation", matrix)
        layer.impl._mxfp_rotation_mode = self.c4_config["mode"]
        layer.impl.q_quant_options = {k: self.c4_config["q"][k] for k in ("block_size", "scale_alg", "dst_type_max")}
        layer.impl.k_quant_max = self.c4_config["k"]["dst_type_max"]
        layer.impl.qfa_quant_schema = self.c4_config["quant_schema"]

    def process_weights_after_loading(self, layer):
        if self.c4_config is None:
            return super().process_weights_after_loading(layer)
        raw = layer.v_cache_scale.detach().cpu()
        if ((raw == 0) | (raw == 255)).any():
            raise ValueError("C4 checkpoint V scale was not loaded or has invalid E8M0 values")
        matrix = layer._mxfp_rotation.detach().cpu().contiguous()
        matrix_hash = hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
        if matrix_hash != self.c4_config["matrix_sha256"]:
            raise ValueError("C4 checkpoint rotation is missing or does not match metadata SHA256")
        super().process_weights_after_loading(layer)
        layer.register_buffer("_mxfp_inverse_rotation", layer._mxfp_rotation.detach().T.contiguous(), persistent=False)
        layer._mxfp_checkpoint_loaded = True
        logger.info(
            "C4 checkpoint loaded layer=%s V_elements=%d V_sha256=%s rotation_sha256=%s Q_alg=2 K_alg=2 max=7.25 schema=2",
            self.prefix, raw.numel(), hashlib.sha256(raw.numpy().tobytes()).hexdigest(), matrix_hash,
        )
