# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EPILOGUE_HEADER = (
    ROOT
    / "csrc/mc2/mega_moe/op_kernel/arch35/"
    "block_epilogue_swiglu_mx_quant.h"
)


def _epilogue_source() -> str:
    return EPILOGUE_HEADER.read_text(encoding="utf-8")


def test_non_interleaved_mx_scale_uses_padded_gm_copy():
    source = _epilogue_source()

    required_contract = (
        "quantScaleBlockOutput_",
        "TransMxScaleLayout(singleM_, scaleBlockN_)",
        "CopyScaleFromUb2Gm(singleM_, yScaleOffset, "
        "quantScaleBlockOutput_)",
    )
    for snippet in required_contract:
        assert snippet in source, f"missing padded MX-scale copy step: {snippet}"


def test_mx_scale_debug_covers_compute_layout_and_gm_copy():
    source = _epilogue_source()

    required_probes = (
        "clamp limit",
        "SwiGLU result UB before MX quant",
        "max exponent UB after ComputeMaxExp",
        "half scale UB after ComputeScale",
        "quant data UB after MX quant",
        "scale compact UB after ComputeScale",
        "scale padded UB after TransMxScaleLayout",
        "scale GM after DataCopyPad",
    )
    for probe in required_probes:
        assert probe in source, f"missing one-pass MX-scale probe: {probe}"

    assert "SyncFuncStatic<AscendC::HardEvent::V_S" in source
    assert "SyncFuncStatic<AscendC::HardEvent::S_V" in source
