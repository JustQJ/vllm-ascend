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
 * \file mega_moe_workspace_info.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__ [aicore]
#else
#define GM_ADDR uint8_t*
#define HOST_DEVICE
#endif

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
constexpr int32_t MAX_AICORE_NUM = 36;
// wave-grain dispatch flag: must match MegaMoeImpl::L1_TILE_M
constexpr int64_t DISPATCH_WAVE_TILE_M = 256;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t FP4_NUM = 2U;
} // namespace

struct WorkspaceInfo {
    GM_ADDR gmCtx;
    GM_ADDR ptrA;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;
    GM_ADDR ptrcumsumMM;
    GM_ADDR expandedRowIdx;
    GM_ADDR expandedRowIdxGather;
    GM_ADDR ptrSumBeforeRank;
    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrFlagDispatchToGmm1;
    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        workspaceSize = 0;
        ptrA = base;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        ptrAScale = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h /
            MXFP_SCALE_GROUP_NUM, ALIGN_512);
        ptrA2 = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize *
            tilingData->hiddenDim / SWIGLU_N_HALF, ALIGN_512);
        ptrA2Scale = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim /
            SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM, ALIGN_512);
        ptrcumsumMM = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize *
            tilingData->expertPerRank, ALIGN_512);
        expandedRowIdx = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->bs), ALIGN_256) *
            tilingData->topK;
        ptrSumBeforeRank = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) *
            tilingData->expertPerRank, ALIGN_128);
        ptrFlagSwiGluToGmm2 = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE;
        ptrFlagDispatchToGmm1 = base + workspaceSize;
        // wave-grain dispatch flag: per expert allocate one slot per wave (L1_TILE_M=256 rows),
        // aligned up to INT_CACHELINE so each expert's slot block stays cache-line clean.
        int64_t maxWavesPerExpert = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize),
            DISPATCH_WAVE_TILE_M);
        int64_t dispatchFlagSlotsPerExpert = Ops::Base::CeilAlign(maxWavesPerExpert,
            static_cast<int64_t>(INT_CACHELINE));
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * dispatchFlagSlotsPerExpert;
        expandedRowIdxGather = ptrA;
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H