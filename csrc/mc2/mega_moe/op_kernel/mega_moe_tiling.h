/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tiling.h
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING
#define ASCENDC_MEGA_MOE_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_arch35_tiling_def.h"

using namespace AscendC;

struct MegaMoeTilingData {
    MoeInitRoutingV3Arch35TilingData moeInitRoutingTilingData;
    uint32_t expertPerRank;
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP; // if blockNumPerEP < dispatchRows, then dispatchRows is invalid
    uint32_t maxOutputSize;
    uint32_t rankId;
    uint32_t topK;
    int32_t dispatchRows;
    uint8_t groupListType;
    bool transX;
    bool transW;
    bool transW2;
};
#endif