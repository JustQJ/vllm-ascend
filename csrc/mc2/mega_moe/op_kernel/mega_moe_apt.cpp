/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif

#include "kernel_operator.h"

#ifdef ENABLE_TENSOR_API
#include "mega_moe.h"
#include "mega_moe_layered.h"
#endif

#include "mega_moe_tiling.h"
#include "mega_moe_tiling_key.h"

using namespace AscendC;
#ifdef ENABLE_TENSOR_API
using namespace MegaMoeImpl;
#endif

template<uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType, uint8_t CommModeType>
__global__ __aicore__ void mega_moe(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR xActiveMask,
    GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
// ── arch35 MegaMoe entry/exit diagnostic ──────────────────────────────────────
// 0 = release default (no prints); 1 = trace ENTER/EXIT around arch35 dispatch
// Used for narrowing down OP_2026_07_10_001 (y all-zero on 950 EP_2).
#ifndef MEGA_MOE_ARCH35_DEBUG
#define MEGA_MOE_ARCH35_DEBUG 1
#endif
#if MEGA_MOE_ARCH35_DEBUG
// Ascend C kernel path: device-side compilation; use AscendC::printf (host stdio
// is not available on the AICore). <kernel_operator.h> already pulls in the
// declaration, so we deliberately do NOT include <cstdio> here.
//
// We only print a static string (no %u/%p) because some CANN toolkits reject
// format arguments in kernel-side printf.  The ENTER/EXIT tag alone tells us
// which branch (MTE / URMA / non-MXFP) was selected.
#define MM_DBG_ENTER()                                                                 \
    do {                                                                               \
        AscendC::printf("[MMDBG] arch35 MEGA_MOE_ENTER\n");                           \
    } while (0)
#define MM_DBG_EXIT(tag)                                                               \
    do {                                                                               \
        AscendC::printf("[MMDBG] arch35 MEGA_MOE_EXIT " tag "\n");                    \
    } while (0)
#else
#define MM_DBG_ENTER() ((void)0)
#define MM_DBG_EXIT(tag) ((void)0)
#endif
// ──────────────────────────────────────────────────────────────────────────────

#if defined(ENABLE_TENSOR_API) && \
    defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16) && \
    defined(ORIG_DTYPE_Y) && (ORIG_DTYPE_Y == DT_BF16) && \
    defined(ORIG_DTYPE_WEIGHT1) && \
        ((ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E5M2) || \
         (ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E4M3FN) || \
         (ORIG_DTYPE_WEIGHT1 == DT_FLOAT4_E2M1)) && \
    defined(ORIG_DTYPE_WEIGHT2) && (ORIG_DTYPE_WEIGHT2 == ORIG_DTYPE_WEIGHT1)
    if constexpr (CommModeType == TILINGKEY_TPL_MTE) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            MM_DBG_ENTER();
            MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1,
                DispatchQuantOutType, CombineQuantOutType> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
            MM_DBG_EXIT("MTE");
        } else {
            MM_DBG_EXIT("MTE-NON-MXFP");
        }
    } else if constexpr (CommModeType == TILINGKEY_TPL_URMA) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            MM_DBG_ENTER();
            MegaMoeLayered<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1,
                DispatchQuantOutType, CombineQuantOutType> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
            MM_DBG_EXIT("URMA");
        } else {
            MM_DBG_EXIT("URMA-NON-MXFP");
        }
    }
#endif
}