# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import torch
from vllm.distributed import get_tp_group

from vllm_ascend import envs

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class MLADataCollectorConfig:
    dump_dir: Path | None
    max_tokens: int = 128
    max_steps: int = 1
    layers: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls) -> MLADataCollectorConfig:
        dump_dir = envs.VLLM_ASCEND_MLA_DATA_DUMP_DIR
        layer_filter = frozenset(
            item.strip() for item in envs.VLLM_ASCEND_MLA_DATA_DUMP_LAYERS.split(",") if item.strip()
        )
        return cls(
            dump_dir=Path(dump_dir).expanduser() if dump_dir else None,
            max_tokens=envs.VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS,
            max_steps=envs.VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS,
            layers=layer_filter,
        )

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("VLLM_ASCEND_MLA_DATA_DUMP_MAX_TOKENS must be positive")
        if self.max_steps <= 0:
            raise ValueError("VLLM_ASCEND_MLA_DATA_DUMP_MAX_STEPS must be positive")


class MLADataCollector:
    """Persist bounded MLA tensor samples for offline distribution analysis.

    Tensor copies intentionally synchronize the device. The collector is
    therefore disabled unless ``VLLM_ASCEND_MLA_DATA_DUMP_DIR`` is set.
    """

    def __init__(self, layer_name: str | None, config: MLADataCollectorConfig | None = None) -> None:
        self.layer_name = layer_name or "unknown_layer"
        self.config = config or MLADataCollectorConfig.from_env()
        self._steps: dict[str, int] = {}
        self._lock = threading.Lock()

        if self.enabled and self._tp_rank() == 0:
            assert self.config.dump_dir is not None
            self.config.dump_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.config.dump_dir is not None and self._layer_is_enabled()

    def _layer_is_enabled(self) -> bool:
        if not self.config.layers:
            return True
        if self.layer_name in self.config.layers:
            return True
        match = _LAYER_INDEX_PATTERN.search(self.layer_name)
        return match is not None and match.group(1) in self.config.layers

    @staticmethod
    def _global_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return get_tp_group().rank
        return int(os.getenv("RANK", "0"))

    @staticmethod
    def _tp_rank() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return get_tp_group().rank_in_group
        return 0

    @staticmethod
    def _tp_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return get_tp_group().world_size
        return 1

    def capture(
        self,
        phase: str,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any] | None = None,
        tp_sharded_dims: dict[str, int] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        with self._lock:
            step = self._steps.get(phase, 0)
            if step >= self.config.max_steps:
                return None
            self._steps[phase] = step + 1

        tp_sharded_dims = tp_sharded_dims or {}
        tp_size = self._tp_size()
        tp_rank = self._tp_rank()
        gathered_tensors = {}
        tensor_metadata = {}
        for name, tensor in tensors.items():
            sample = tensor.detach()
            if sample.ndim > 0:
                sample = sample[: self.config.max_tokens]
            sample = sample.contiguous()
            gather_dim = tp_sharded_dims.get(name)
            if gather_dim is not None and tp_size > 1:
                sample = get_tp_group().all_gather(sample, dim=gather_dim)
            tensor_metadata[name] = {
                "original_shape": tuple(tensor.shape),
                "saved_shape": tuple(sample.shape),
                "dtype": str(tensor.dtype),
                "tp_gather_dim": gather_dim,
            }
            gathered_tensors[name] = sample

        if tp_rank != 0:
            return None

        cpu_tensors = {name: tensor.cpu() for name, tensor in gathered_tensors.items()}

        global_rank = self._global_rank()
        safe_layer_name = _SAFE_FILENAME_PATTERN.sub("_", self.layer_name).strip("_")
        filename = f"{safe_layer_name}.{phase}.rank{global_rank}.pid{os.getpid()}.step{step:04d}.pt"
        assert self.config.dump_dir is not None
        output_path = self.config.dump_dir / filename
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        payload = {
            "format_version": 2,
            "layer_name": self.layer_name,
            "phase": phase,
            "rank": global_rank,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "step": step,
            "metadata": metadata or {},
            "tensor_metadata": tensor_metadata,
            "tensors": cpu_tensors,
        }
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
        return output_path
