// SPDX-License-Identifier: Apache-2.0
#include "turbo_quant_group_prepare_tiling.h"
namespace optiling {
static ge::graphStatus TilingGroupPrepare(gert::TilingContext *context)
{
    const auto *indices = context->GetInputShape(0);
    const auto *qLengths = context->GetInputShape(1);
    const auto *kvLengths = context->GetInputShape(2);
    if (indices == nullptr || qLengths == nullptr || kvLengths == nullptr ||
        context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = indices->GetStorageShape();
    const auto &q = qLengths->GetStorageShape();
    const auto &kv = kvLengths->GetStorageShape();
    const auto *mode = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (shape.GetDimNum() != 3 || shape.GetDim(0) <= 0 || shape.GetDim(0) > INT32_MAX ||
        shape.GetDim(1) != 1 || shape.GetDim(2) <= 0 || shape.GetDim(2) > tq_group::MAX_TOPK ||
        q.GetDimNum() != 1 || kv.GetDimNum() != 1 || q.GetDim(0) <= 0 ||
        q.GetDim(0) > INT32_MAX || q.GetDim(0) != kv.GetDim(0) ||
        mode == nullptr || (*mode != 0 && *mode != 3)) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        const auto *desc = context->GetInputDesc(i);
        if (desc == nullptr || desc->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // raw 32 KiB + hash 64 KiB + IDs 12 KiB + owners 3 KiB + descriptors.
    constexpr uint64_t REQUIRED_UB = (tq_group::MAX_GROUP * tq_group::MAX_TOPK +
        2 * tq_group::HASH_SIZE + tq_group::QUAD_CAP + tq_group::MAX_GROUP * tq_group::DESC_WORDS) *
        sizeof(int32_t) + tq_group::QUAD_CAP;
    if (ubBytes < REQUIRED_UB) {
        return ge::GRAPH_FAILED;
    }
    TurboQuantGroupPrepareTilingData data;
    data.set_tokens(shape.GetDim(0));
    data.set_topk(shape.GetDim(2));
    data.set_batches(q.GetDim(0));
    data.set_sparseMode(*mode);
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || workspace == nullptr || raw->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    const uint32_t cores = platform.GetCoreNumAiv();
    context->SetBlockDim(cores == 0 ? 1 : cores);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(TurboQuantGroupPrepare).Tiling(TilingGroupPrepare);
} // namespace optiling
