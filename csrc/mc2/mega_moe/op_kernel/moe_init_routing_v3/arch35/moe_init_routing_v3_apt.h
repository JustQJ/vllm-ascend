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
 * \file moe_init_routing_v3_apt.cpp
 * \brief
 */
#include "moe_v3_mrgsort_out.h"
#include "moe_v3_mrgsort.h"
#include "moe_v3_sort_one_core.h"
#include "moe_v3_sort_multi_core.h"
#include "moe_v3_expert_tokens_count.h"
#include "moe_v3_row_idx_gather.h"
#include "moe_v3_gather_out.h"
#include "moe_v3_gather_dynamic_quant.h"
#include "moe_v3_gather_mxfp8_quant.h"
#include "moe_v3_gather_mxfp4_quant.h"

/*
 * 非量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_GATHER 1000000    // 单核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER 1001000   // 单核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER 1100000  // 多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER 1101000 // 多核排序、非量化、SCATTER索引

/*
 * 动态量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER 1020000    // 单核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER 1021000   // 单核排序、动态量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER 1120000  // 多核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER 1121000 // 多核排序、动态量化、SCATTER索引

/*
 * MXFP8量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER 1030000    // 单核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER 1031000   // 单核排序、MXFP8量化、SCATTER索引 // 小数据量
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER 1130000  // 多核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER 1131000 // 多核排序、MXFP8量化、SCATTER索引 // 大数据量

using namespace AscendC;
using namespace MoeInitRoutingV3;
template<typename DTYPE_X, typename DTYPE_EXPANDED_X>
__aicore__ inline void moe_init_routing_v3(
        GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR expandedX, GM_ADDR expandedRowIdx,
        GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
        GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData* tilingData, uint64_t tilingKey)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    auto t = &tilingData;
    
    // 1.排序阶段，计算SortedExpertIdx和SortedRowIdx。若rowIdxType=1(Scatter)，则SortedRowIdx直接写到输出expandedRowIdx。
    TPipe sortPipe;
    if (tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_GATHER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) {
        // 单核排序
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, workspace, tilingData, &sortPipe);
        op.Process();
    } else if (tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER) {
        // 多核排序
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, workspace, tilingData, &sortPipe);
        op.Process();
    }
    
    sortPipe.Destroy();
    // 2.TokensCount阶段，计算输出expertTokensCountOrCumsum
    TPipe histogramPipe;
    ExpertTokensCount countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, &histogramPipe);
    countOp.Process();
    histogramPipe.Destroy();

    // 3.若rowIdxType=0(Gather)，映射计算输出expandedRowIdx；否则该输出在阶段1就被写出

    // 4.直接搬运或是搬运的过程中对x进行量化
    if (std::is_same_v<DTYPE_EXPANDED_X, fp4x2_e2m1_t>) {
        TPipe gatherPipe;
        MoeV3GatherMxfp4Quant<DTYPE_X, DTYPE_EXPANDED_X> gatherMxfp4QuantOp;
        gatherMxfp4QuantOp.Init(x, scale, workspace, expandedRowIdx, expandedX, expandedScale, tilingData, &gatherPipe);
        gatherMxfp4QuantOp.Process();
        gatherPipe.Destroy();
    } else {
        TPipe gatherPipe;
        MoeGatherOutMxfp8Quant<DTYPE_X, DTYPE_EXPANDED_X> gatherMxfp8QuantOp;
        gatherMxfp8QuantOp.Init(x, scale, workspace, expandedRowIdx, expandedX, expandedScale, tilingData, &gatherPipe);
        gatherMxfp8QuantOp.Process();
        gatherPipe.Destroy();
    }
}
