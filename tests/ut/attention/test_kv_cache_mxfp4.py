# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_c4_mxfp import mxfp4_cache_spec, mxfp4_cache_views_for_spec
from vllm_ascend.attention.attention_c8_mxfp import fill_mxfp_v_scale_cache


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("ratio", [1, 8])
@pytest.mark.parametrize("padding", [0, 256])
def test_c4_packed_budget_and_nonzero_offset(dim, heads, ratio, padding):
    spec = mxfp4_cache_spec(
        FullAttentionSpec(
            block_size=512 * ratio, num_kv_heads=heads, head_size=dim, head_size_v=dim, dtype=torch.bfloat16
        )
    )
    assert spec.state_content_bytes == dim + dim // 16
    assert spec.page_size_bytes == 512 * ratio * heads * (dim + dim // 16)
    spec = replace(spec, page_size_padded=spec.page_size_bytes + padding * ratio)
    backing = torch.full((128 + 2 * spec.page_size_bytes + 64,), 99, dtype=torch.int8)
    raw = backing[128:-64]
    k, v, ks, vs = mxfp4_cache_views_for_spec(raw, spec, 512)
    assert k.shape == (2 * ratio, 512, heads, dim // 2)
    assert ks.shape == (2 * ratio, 512, heads, dim // 64, 2)
    assert vs.shape == (2 * ratio, 8, heads, dim, 2)
    pitch = spec.page_size_bytes // ratio
    start = 0
    pages = raw.view(2 * ratio, pitch)
    for marker, view in enumerate((k, ks, v, vs), start=1):
        size = view[0].numel()
        assert view.stride(0) == pitch and view[0].is_contiguous()
        view.fill_(marker)
        assert (pages[:, start : start + size] == marker).all()
        start += size
    assert (pages[:, start:] == 99).all()
    assert (backing[:128] == 99).all() and (backing[-64:] == 99).all()


def test_c4_fill_selected_subview_preserves_other_scale_pages():
    spec = mxfp4_cache_spec(
        FullAttentionSpec(block_size=1024, num_kv_heads=2, head_size=256, head_size_v=256, dtype=torch.bfloat16)
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    _, _, _, vs = mxfp4_cache_views_for_spec(raw, spec, 512)
    scale = torch.arange(512).remainder(200).to(torch.uint8)
    fill_mxfp_v_scale_cache(scale, vs)
    expected = vs.clone()
    # This utility is only safe when the caller already knows the owner.
    vs[2:4].zero_()
    fill_mxfp_v_scale_cache(scale, vs[2:4])
    assert torch.equal(vs, expected)


def test_c4_and_c8_metadata_plans_do_not_alias():
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from vllm_ascend.attention.attention_c4_mxfp import AscendC4MXFPAttentionBackendImpl
    from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl

    metadata = SimpleNamespace()
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_kv=torch.tensor([33], dtype=torch.int32),
        max_seqlen_q=1,
        mask_mode=3,
        layout_q_descale="N2TGD",
    )
    op = MagicMock(side_effect=[object(), object()])
    plans = []
    with patch("vllm_ascend.attention.mxfp_static.qfa_ops", return_value=SimpleNamespace(quant_flash_attn_metadata=op)):
        for cls in (AscendC8MXFPAttentionBackendImpl, AscendC4MXFPAttentionBackendImpl):
            impl = object.__new__(cls)
            impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 1, 256
            impl.vllm_config = SimpleNamespace(additional_config={})
            plans.append(impl._get_qfa_metadata(metadata, **kwargs))
    assert plans[0] is not plans[1]
    assert [call.args[3] for call in op.call_args_list] == [1, 5]
    assert [call.kwargs["layout_kv"] for call in op.call_args_list] == ["PA_NZ", "PA_BBND"]


@pytest.mark.parametrize("heads", [1, 2])
def test_c4_hybrid_planning_counts_packed_one_token_page(heads):
    from vllm_ascend.attention.attention_c4_mxfp import AscendC4MXFPAttentionBackend

    spec = AscendC4MXFPAttentionBackend.customize_spec(
        FullAttentionSpec(block_size=1, num_kv_heads=heads, head_size=256, dtype=torch.uint8)
    )
    assert spec.page_size_bytes == heads * 272
    assert spec.head_size == 256 and spec.head_size_v == 256
    assert AscendC4MXFPAttentionBackend.get_supported_kernel_block_sizes() == [512]


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
def test_packet_target_and_bf16_draft_keep_logical_blocks(mode):
    from vllm_ascend.attention.attention_c8_mxfp import mxfp_cache_spec
    from vllm_ascend.worker.v2.attn_utils import _align_hybrid_attention_page_sizes

    dense = FullAttentionSpec(block_size=4096, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    target = mxfp4_cache_spec(dense) if mode == "mxfp4" else mxfp_cache_spec(dense)
    draft = replace(dense, is_draft=True)
    assert draft.real_page_size_bytes % target.real_page_size_bytes != 0
    specs = {"target": target, "draft": draft}
    _align_hybrid_attention_page_sizes(specs)
    assert specs == {"target": target, "draft": draft}


def test_packet_and_bf16_views_do_not_write_other_physical_pages():
    from vllm_ascend.worker.v2.attn_utils import _reshape_combined_attention_kv_cache

    dense = FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    spec = replace(mxfp4_cache_spec(dense), page_size_padded=dense.page_size_bytes)
    pitch = spec.page_size_bytes
    raw = torch.full((3 * pitch,), 19, dtype=torch.int8)
    packet = mxfp4_cache_views_for_spec(raw, spec, 512)
    k, v = _reshape_combined_attention_kv_cache(raw, (2, 3, 512, 1, 256), torch.bfloat16, pitch)
    for code, part in enumerate(packet):
        part[1].fill_(code + 1)
    written_packet = raw[pitch : 2 * pitch].clone()
    k[2].fill_(11)
    v[2].fill_(12)
    assert torch.all(raw[:pitch] == 19)
    assert torch.equal(raw[pitch : 2 * pitch], written_packet)
    assert torch.all(raw[pitch + spec.real_page_size_bytes : 2 * pitch] == 19)
    assert torch.all(k[2] == 11) and torch.all(v[2] == 12)
    for code, part in enumerate(packet):
        assert torch.all(part[1] == code + 1)


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
def test_merged_bf16_draft_is_not_packet_cache(mode, monkeypatch):
    from types import SimpleNamespace

    from vllm_ascend.attention.attention_c8_mxfp import mxfp_cache_spec
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.utils import is_mxfp_kv_quant
    from vllm_ascend.worker.v2 import attn_utils

    dense = FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=256, dtype=torch.bfloat16, is_draft=True)
    merged = FullAttentionSpec.merge([dense])
    assert not merged.is_draft  # Actual upstream merge drops this ownership hint.
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=mode),
        speculative_config=SimpleNamespace(kv_cache_dtype="auto"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text")),
    )
    assert not is_mxfp_kv_quant(config, merged)
    target = (
        mxfp4_cache_spec(replace(dense, is_draft=False))
        if mode == "mxfp4"
        else mxfp_cache_spec(replace(dense, is_draft=False))
    )
    assert is_mxfp_kv_quant(config, FullAttentionSpec.merge([target]))
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    group = SimpleNamespace(
        kv_cache_group_id=0, kv_cache_spec=merged, layer_names=["draft"], backend=AscendAttentionBackend
    )
    cache_config = SimpleNamespace(num_blocks=2, kv_cache_groups=[group], kv_cache_tensors=[])
    raw = torch.zeros(2 * merged.page_size_bytes, dtype=torch.int8)
    caches = attn_utils._reshape_kv_cache_v2([group], {"draft": raw}, mode, [512], {}, cache_config)
    assert len(caches["draft"]) == 2
    assert all(part.dtype == torch.bfloat16 for part in caches["draft"])
    assert all(part.shape == (2, 512, 1, 256) for part in caches["draft"])
