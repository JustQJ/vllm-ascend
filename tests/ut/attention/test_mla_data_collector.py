from pathlib import Path
from unittest.mock import patch

import torch

from vllm_ascend.attention.mla_data_collector import MLADataCollector, MLADataCollectorConfig


def test_collector_is_disabled_without_dump_dir():
    collector = MLADataCollector(
        "model.layers.0.self_attn.attn",
        MLADataCollectorConfig(dump_dir=None),
    )

    assert not collector.enabled
    assert collector.capture("prefill", {"q_latent": torch.ones(2, 3)}) is None


def test_collector_filters_layers(tmp_path: Path):
    config = MLADataCollectorConfig(dump_dir=tmp_path, layers=frozenset({"3"}))

    assert not MLADataCollector("model.layers.2.self_attn.attn", config).enabled
    assert MLADataCollector("model.layers.3.self_attn.attn", config).enabled


def test_collector_bounds_tokens_and_steps(tmp_path: Path):
    collector = MLADataCollector(
        "model.layers.7.self_attn.attn",
        MLADataCollectorConfig(dump_dir=tmp_path, max_tokens=2, max_steps=1),
    )
    tensor = torch.arange(12, dtype=torch.float32).view(4, 3)

    output_path = collector.capture("decode", {"q_latent": tensor}, metadata={"kv_lora_rank": 512})

    assert output_path is not None
    payload = torch.load(output_path, weights_only=False)
    assert payload["format_version"] == 2
    assert payload["layer_name"] == "model.layers.7.self_attn.attn"
    assert payload["metadata"] == {"kv_lora_rank": 512}
    assert torch.equal(payload["tensors"]["q_latent"], tensor[:2])
    assert payload["tensor_metadata"]["q_latent"]["original_shape"] == (4, 3)
    assert payload["tensor_metadata"]["q_latent"]["saved_shape"] == (2, 3)
    assert collector.capture("decode", {"q_latent": tensor}) is None
    assert output_path.parent == tmp_path / "model.layers.7.self_attn.attn"
    assert output_path.name.startswith("decode.rank0.pid")
    assert len(list(tmp_path.rglob("*.pt"))) == 1


class _FakeTPGroup:
    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = 2

    @staticmethod
    def all_gather(tensor: torch.Tensor, dim: int):
        return torch.cat((tensor, tensor + 10), dim=dim)


@patch("vllm_ascend.attention.mla_data_collector.torch.distributed.is_initialized", return_value=True)
def test_collector_gathers_q_heads_and_only_tp_rank_zero_saves(_is_initialized, tmp_path: Path):
    q_local = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    q_nope_prefill = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    kv_replicated = torch.arange(6, dtype=torch.float32).view(2, 1, 3)
    tensors = {
        "q_latent": q_local,
        "q_nope_prefill": q_nope_prefill,
        "kv_latent": kv_replicated,
    }
    tp_sharded_dims = {"q_latent": 1, "q_nope_prefill": 1}

    with patch(
        "vllm_ascend.attention.mla_data_collector.get_tp_group",
        return_value=_FakeTPGroup(rank=1),
    ):
        non_root = MLADataCollector("model.layers.7.self_attn.attn", MLADataCollectorConfig(dump_dir=tmp_path))
        assert non_root.capture("prefill", tensors, tp_sharded_dims=tp_sharded_dims) is None
        assert not list(tmp_path.rglob("*.pt"))

    with patch(
        "vllm_ascend.attention.mla_data_collector.get_tp_group",
        return_value=_FakeTPGroup(rank=0),
    ):
        root = MLADataCollector("model.layers.7.self_attn.attn", MLADataCollectorConfig(dump_dir=tmp_path))
        output_path = root.capture("prefill", tensors, tp_sharded_dims=tp_sharded_dims)

    assert output_path is not None
    assert output_path.parent == tmp_path / "model.layers.7.self_attn.attn"
    payload = torch.load(output_path, weights_only=False)
    assert payload["tp_rank"] == 0
    assert payload["tp_size"] == 2
    assert torch.equal(payload["tensors"]["q_latent"], torch.cat((q_local, q_local + 10), dim=1))
    assert torch.equal(
        payload["tensors"]["q_nope_prefill"],
        torch.cat((q_nope_prefill, q_nope_prefill + 10), dim=1),
    )
    assert torch.equal(payload["tensors"]["kv_latent"], kv_replicated)
    assert payload["tensor_metadata"]["q_latent"]["tp_gather_dim"] == 1
    assert payload["tensor_metadata"]["q_nope_prefill"]["tp_gather_dim"] == 1
    assert payload["tensor_metadata"]["kv_latent"]["tp_gather_dim"] is None


def test_collector_validates_bounds():
    for field in ("max_tokens", "max_steps"):
        kwargs = {field: 0}
        try:
            MLADataCollectorConfig(dump_dir=Path("/tmp/mla-dump"), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {field}")
