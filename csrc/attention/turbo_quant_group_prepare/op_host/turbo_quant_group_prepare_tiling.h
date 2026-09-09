// SPDX-License-Identifier: Apache-2.0
#ifndef TURBO_QUANT_GROUP_PREPARE_TILING_H
#define TURBO_QUANT_GROUP_PREPARE_TILING_H
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../turbo_quant_sparse_flash_attention/turbo_quant_group_plan.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TurboQuantGroupPrepareTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokens);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, batches);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TurboQuantGroupPrepare, TurboQuantGroupPrepareTilingData)
struct TurboQuantGroupPrepareCompileInfo {};
} // namespace optiling
#endif
