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
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"
#include "platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "hcom_topo_info.h"
#include "mega_moe_tiling.h"
#include "../op_kernel/mega_moe_ceil_utils.h"
#include "../op_kernel/mega_moe_tiling.h"
#include "../op_kernel/mega_moe_tiling_key.h"
#include "../op_kernel/mega_moe_workspace_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
    // init routing
    const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;
    const static int64_t SORT_API_MAX_ELEM = 32 * 255LL;
    const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
    const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

    const static int64_t NUM_TWO = 2LL;
    const static int64_t NUM_FOUR = 4LL;
    const static int64_t MRG_LIST_NUM = 4LL;
    const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
    const static int64_t UB_BLOCK_SIZE = 32LL;
    const static size_t DIM_TWO = 2ULL;
    const static int64_t KV_FACTOR = 2LL;
    const static int64_t EXPERT_IDX_MAX = 10240LL;
    const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

    const static int64_t ROW_IDX_GATHER = 0LL;
    const static int64_t ROW_IDX_SCATTER = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
    const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;
    const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;

    const static int64_t FOUR_DIMS = 4LL;
    const static int64_t THREE_DIMS = 3LL;
    const static int64_t TWO_DIMS = 2LL;
    const static int64_t ONE_DIM = 1LL;
    const static int64_t MIN_BS = 1LL;
    const static int64_t MAX_BS = 512LL;
    const static int64_t MIN_EXPERT_PER_RANK = 1LL;
    const static int64_t MAX_EXPERT_PER_RANK = 16LL;
    const static int64_t H_BASE = 1024LL;
    const static int64_t HIDDEN_DIM_BASE = 1024LL;
    const static int64_t MIN_EP_WORLD_SIZE = 2LL;
    const static int64_t MAX_EP_WORLD_SIZE = 768LL;
    const static int64_t MAX_MOE_EXPERT_NUM = 1024LL;
    const static int64_t DISABLE_EXPERT_CAPACITY = -1LL;
    const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
    const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
}

struct MegaMoeAuxTilingContext {
    // Platform
    int64_t aivCoreNum;
    int64_t totalUbSize;
    int64_t availUbSize;

    // Critical Variable
    int64_t sortLoopMaxElement;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols;
    int64_t inputXDtypeSize;
    int64_t isInputScale;
    int64_t isInputOffset;
    int64_t sortMode;

    // Op Attr
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensNumType;
    int64_t quantMode;
    int64_t expertStart;
    int64_t expertEnd;
    int64_t rowIdxType;

    bool expertTokensNumFlag;
};

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 1) return 0;
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

void PrintMoeV3Arch35VBSComputeTilingData(
    const MoeV3Arch35VBSComputeTilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VBSComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35VMSMiddleComputeTilingData(
    const MoeV3Arch35VMSMiddleComputeTilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35VMSMiddleComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
}

void PrintMoeV3Arch35SortOutComputeTilingData(
    const MoeV3Arch35SortOutComputeTilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35SortOutComputeTilingData ==========");
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35ExpertTokensCountTilingData(
    const MoeV3Arch35ExpertTokensCountTilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35ExpertTokensCountTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
}

void PrintMoeV3Arch35GatherOutComputeTilingData(
    const MoeV3Arch35GatherOutComputeTilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeV3Arch35GatherOutComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreIndicesElements is %ld", data.perCoreIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesElements is %ld", data.lastCoreIndicesElements);
    OP_LOGD(nodeName, "perCoreIndicesLoops is %ld", data.perCoreIndicesLoops);
    OP_LOGD(nodeName, "perCorePerLoopIndicesElements is %ld", data.perCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "perCoreLastLoopIndicesElements is %ld", data.perCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesLoops is %ld", data.lastCoreIndicesLoops);
    OP_LOGD(nodeName, "lastCorePerLoopIndicesElements is %ld", data.lastCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreLastLoopIndicesElements is %ld", data.lastCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "colsLoops is %ld", data.colsLoops);
    OP_LOGD(nodeName, "perLoopCols is %ld", data.perLoopCols);
    OP_LOGD(nodeName, "lastLoopCols is %ld", data.lastLoopCols);
}

void PrintMoeInitRoutingV3Arch35TilingData(
    const MoeInitRoutingV3Arch35TilingData& data, const char *nodeName)
{
    OP_LOGD(nodeName, "========== MoeInitRoutingV3Arch35TilingData ==========");
    OP_LOGD(nodeName, "coreNum is %ld", data.coreNum);
    OP_LOGD(nodeName, "BS is %ld", data.n);
    OP_LOGD(nodeName, "H is %ld", data.cols);
    OP_LOGD(nodeName, "topK is %ld", data.k);
    OP_LOGD(nodeName, "expertStart is %ld", data.expertStart);
    OP_LOGD(nodeName, "expertEnd is %ld", data.expertEnd);
    OP_LOGD(nodeName, "actualExpertNum is %ld", data.actualExpertNum);
    OP_LOGD(nodeName, "quantMode is %ld", data.quantMode);
    OP_LOGD(nodeName, "rowIdxType is %ld", data.rowIdxType);
    OP_LOGD(nodeName, "isInputScale is %s", data.isInputScale ? "True" : "False");
    OP_LOGD(nodeName, "isInputOffset is %s", data.isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "expertNum is %ld", data.expertNum);
    OP_LOGD(nodeName, "expertTokensNumType is %ld", data.expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %ld", data.expertTokensNumFlag);
    OP_LOGD(nodeName, "gatherFirstFullload is %ld", data.gatherFirstFullload);
    OP_LOGD(nodeName, "epFullload is %ld", data.epFullload);
    OP_LOGD(nodeName, "activeNum is %ld", data.activeNum);
    OP_LOGD(nodeName, "dropPadMode is %ld", data.dropPadMode);
    OP_LOGD(nodeName, "smoothType is %ld", data.smoothType);
    OP_LOGD(nodeName, "InitRouting TilingKey is %ld", data.tilingKey);

    PrintMoeV3Arch35VBSComputeTilingData(data.vbsComputeParamsOp, nodeName);
    PrintMoeV3Arch35VMSMiddleComputeTilingData(data.vmsMiddleComputeParamsOp, nodeName);
    PrintMoeV3Arch35SortOutComputeTilingData(data.sortOutComputeParamsOp, nodeName);
    PrintMoeV3Arch35ExpertTokensCountTilingData(data.expertTokensCountTilingDataOp, nodeName);
    PrintMoeV3Arch35GatherOutComputeTilingData(data.gatherOutComputeParamsOp, nodeName);
}

void PrintMegaMoeTilingData(const MegaMoeTilingData* tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    
    PrintMoeInitRoutingV3Arch35TilingData(tilingData->moeInitRoutingTilingData, nodeName);

    OP_LOGD(nodeName, "BS is %u", tilingData->bs);
    OP_LOGD(nodeName, "H is %u", tilingData->h);
    OP_LOGD(nodeName, "hiddenDim is %u", tilingData->hiddenDim);

    OP_LOGD(nodeName, "topK is %u", tilingData->topK);
    OP_LOGD(nodeName, "expertPerRank is %u", tilingData->expertPerRank);
    OP_LOGD(nodeName, "groupListType is %u", tilingData->groupListType);

    OP_LOGD(nodeName, "epWorldSize is %u", tilingData->epWorldSize);
    OP_LOGD(nodeName, "maxOutputSize is %u", tilingData->maxOutputSize);

    OP_LOGD(nodeName, "transX is %s", (tilingData->transX ? "True" : "False"));
    OP_LOGD(nodeName, "transW is %s", (tilingData->transW ? "True" : "False"));
    OP_LOGD(nodeName, "transW2 is %s", (tilingData->transW2 ? "True" : "False"));
}

void printWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "ptrA:                        %ld\n", info->ptrA);
    OP_LOGD(nodeName, "ptrAScale:                   %ld\n", info->ptrAScale);
    OP_LOGD(nodeName, "ptrA2:                       %ld\n", info->ptrA2);
    OP_LOGD(nodeName, "ptrA2Scale:                  %ld\n", info->ptrA2Scale);
    OP_LOGD(nodeName, "ptrcumsumMM:                 %ld\n", info->ptrcumsumMM);
    OP_LOGD(nodeName, "expandedRowIdx:              %ld\n", info->expandedRowIdx);
    OP_LOGD(nodeName, "ptrSumBeforeRank:            %ld\n", info->ptrSumBeforeRank);
    OP_LOGD(nodeName, "ptrFlagSwiGluToGmm2:         %ld\n", info->ptrFlagSwiGluToGmm2);
    OP_LOGD(nodeName, "ptrFlagDispatchToGmm1:       %ld\n", info->ptrFlagDispatchToGmm1);
    OP_LOGD(nodeName, "workspaceSize:               %ld\n", info->workspaceSize);
}

void printPeermemInfo(const MegaMoeTilingData* tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");

    int64_t tokenPerExpert = static_cast<int64_t>(tilingData->epWorldSize) *
        ops::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) *
        tilingData->expertPerRank, ALIGN_128) *
        sizeof(int32_t);

    int64_t quantTokenScale = static_cast<int64_t>(tilingData->bs) *
        tilingData->topK *
        (tilingData->h + tilingData->h / MXFP_SCALE_GROUP_NUM) *
        sizeof(int8_t);

    int64_t combineOut = static_cast<int64_t>(tilingData->bs) * tilingData->topK * tilingData->h * sizeof(int16_t);

    int64_t peermemSize = PEERMEM_DATA_OFFSET + tokenPerExpert + quantTokenScale + combineOut;
    OP_LOGD(nodeName, "peermemSize: {%ld}\n", peermemSize);

    int64_t offset = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "ptrTokenPerExpert: {%ld}\n", offset);

    offset += tokenPerExpert;
    OP_LOGD(nodeName, "ptrA0: {%ld}\n", PEERMEM_DATA_OFFSET);

    offset += quantTokenScale;
    OP_LOGD(nodeName, "ptrD: {%ld}\n", offset);
}

void printAuxiliaryTilingCtx(const MegaMoeAuxTilingContext* ctx,
    const uint32_t expertPerRank, const uint32_t epWorldSize,
    const std::vector<int64_t>& expertIdxShape,
    ge::DataType xDtype, const char *nodeName)
{
    OP_LOGD(nodeName, "========== AuxiliaryTilingCtx ==========");
    OP_LOGD(nodeName, "BS is %ld", ctx->n);
    OP_LOGD(nodeName, "H is %ld", ctx->cols);
    OP_LOGD(nodeName, "topK is %ld", ctx->k);
    OP_LOGD(nodeName, "expertPerRank is %ld", expertPerRank);
    OP_LOGD(nodeName, "epWorldSize is %ld", epWorldSize);

    OP_LOGD(nodeName, "availUbSize is %ld", ctx->availUbSize);
    OP_LOGD(nodeName, "sortLoopMaxElement is %ld", ctx->sortLoopMaxElement);
    OP_LOGD(nodeName, "totalLength is %ld", ctx->totalLength);
    OP_LOGD(nodeName, "activeNum is %ld", ctx->activeNum);
    OP_LOGD(nodeName, "expertCapacity is %ld", ctx->expertCapacity);
    OP_LOGD(nodeName, "expertNum is %ld", ctx->expertNum);

    OP_LOGD(nodeName, "x is %s, inputXDtypeSize = %ld",
        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), ctx->inputXDtypeSize);
    OP_LOGD(nodeName, "isInputScale is %s, isInputOffset is %s",
        ctx->isInputScale ? "True" : "False", ctx->isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "dropPadMode is %ld, expertTokensNumType is %ld",
        ctx->dropPadMode, ctx->expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %s, quantMode is %ld",
        ctx->expertTokensNumFlag ? "True" : "False", ctx->quantMode);
    
    OP_LOGD(nodeName, "expertIdxShape is [%ld, %ld]", expertIdxShape[0], expertIdxShape[1]);
    OP_LOGD(nodeName, "expertRange is [%ld, %ld]", ctx->expertStart, ctx->expertEnd);
    OP_LOGD(nodeName, "rowIdxType is %ld", ctx->rowIdxType);
}

static int64_t OpQuantModeToInitRoutingQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_TYPE_E5M2: return 2LL;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN: return 3LL;
        case DISPATCH_QUANT_OUT_TYPE_E2M1: return 9LL;
        default: return -1LL;
    }
    return -1LL;
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_TYPE_E5M2: return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN: return ge::DT_FLOAT8_E4M3FN;
        case DISPATCH_QUANT_OUT_TYPE_E2M1: return ge::DT_FLOAT4_E2M1;
        default: return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutType == static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E5M2;
    } else if (dispatchQuantOutType == static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E4M3FN;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E2M1;
    }

    return opQuantMode;
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);

    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode);
}

static ge::graphStatus CheckInitTilingData(
    const int64_t expertTokensNumType, int64_t expertNum, const int64_t dropPadMode, const bool expertTokensNumFlag,
    const int64_t quantMode, const int64_t rowIdxType, const std::vector<int64_t> expertRange,
    const std::vector<int64_t> xShape, const std::vector<int64_t> expertIdxShape,
    const char* nodeName)
{
    OP_TILING_CHECK(
        expertTokensNumType != EXPERT_TOKENS_TYPE_COUNT && expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expert_tokens_num_type",
            std::to_string(expertTokensNumType).c_str(),
            (std::string("EXPERT_TOKENS_TYPE_COUNT(") + std::to_string(EXPERT_TOKENS_TYPE_COUNT) +
             ") or EXPERT_TOKENS_TYPE_KEY_VALUE(" + std::to_string(EXPERT_TOKENS_TYPE_KEY_VALUE) + ")").c_str()),
        return ge::GRAPH_FAILED);

    int64_t maxExpertNum = (expertTokensNumType == EXPERT_TOKENS_TYPE_COUNT) ?
                            EXPERT_IDX_MAX : KV_MODE_EXPERT_IDX_MAX;
    OP_TILING_CHECK(
        expertNum <= 0 || expertNum > maxExpertNum,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "moe_expert_num",
            std::to_string(expertNum).c_str(),
            (std::string("[0, ") + std::to_string(maxExpertNum) + "]").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        dropPadMode != DROP_PAD_MODE_DROPLESS,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "drop_pad_mode",
            std::to_string(dropPadMode).c_str(),
            (std::string("DROP_PAD_MODE_DROPLESS(") + std::to_string(DROP_PAD_MODE_DROPLESS) + ")").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !expertTokensNumFlag,
        OP_LOGE(nodeName, "Failed, expertTokensNumFlag must be true."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        quantMode != DISPATCH_QUANT_OUT_TYPE_E5M2 && quantMode != DISPATCH_QUANT_OUT_TYPE_E4M3FN &&
        quantMode != DISPATCH_QUANT_OUT_TYPE_E2M1,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatch_quant_out_type",
            std::to_string(quantMode).c_str(),
            (std::string("E5M2(") + std::to_string(DISPATCH_QUANT_OUT_TYPE_E5M2) +
             "), E4M3FN(" + std::to_string(DISPATCH_QUANT_OUT_TYPE_E4M3FN) +
             ") or E2M1(" + std::to_string(DISPATCH_QUANT_OUT_TYPE_E2M1) + ")").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        rowIdxType != ROW_IDX_SCATTER && rowIdxType != ROW_IDX_GATHER,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "row_idx_type",
            std::to_string(rowIdxType).c_str(),
            (std::string("ROW_IDX_SCATTER(") + std::to_string(ROW_IDX_SCATTER) +
             ") or ROW_IDX_GATHER(" + std::to_string(ROW_IDX_GATHER) + ")").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertRange.size() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPESIZE(nodeName, "expert_range",
            std::to_string(expertRange.size()).c_str(), "2"),
        return ge::GRAPH_FAILED);

    int64_t expertStart = expertRange[0];
    int64_t expertEnd = expertRange[1];
    OP_TILING_CHECK(
        expertStart < 0 || expertStart >= expertEnd || expertEnd > expertNum,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expert_range",
            (std::string("[") + std::to_string(expertStart) + ", " + std::to_string(expertEnd) + "]").c_str(),
            (std::string("[0, ") + std::to_string(expertNum) + "]").c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xShape.size() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x",
            (std::to_string(xShape.size()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertIdxShape.size() != TWO_DIMS || expertIdxShape[0] != xShape[0],
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "topk_ids and x",
            (std::string("[") + std::to_string(expertIdxShape.size()) + "D, dim0=" +
             std::to_string(expertIdxShape[0]) + "]").c_str(),
            (std::string("topk_ids should be 2D, dim0 should equal x dim0=") + std::to_string(xShape[0])).c_str()),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInitTilingData(
    MegaMoeTilingData &tilingData, MoeInitRoutingV3Arch35TilingData &initTilingData,
    const uint32_t aivNum, const int64_t opQuantMode, const char *nodeName)
{
    // Get Param
    auto bs = tilingData.bs;
    auto h = tilingData.h;
    auto topK = tilingData.topK;
    auto expertPerRank = tilingData.expertPerRank;
    auto epWorldSize = tilingData.epWorldSize;

    int64_t expertTokensNumType = EXPERT_TOKENS_TYPE_COUNT;
    int64_t expertNum = epWorldSize * expertPerRank;
    int64_t dropPadMode = DROP_PAD_MODE_DROPLESS;
    bool expertTokensNumFlag = true;
    int64_t quantMode = opQuantMode;
    int64_t rowIdxType = ROW_IDX_SCATTER;

    const std::vector<int64_t> expertRange = {0, expertNum};
    std::vector<int64_t> xShape = {bs, h};
    std::vector<int64_t> expertIdxShape = {bs, topK};

    bool isInputScale = false;
    bool isInputOffset = false;
    
    // Check Param
    OP_TILING_CHECK(
        CheckInitTilingData(
            expertTokensNumType, expertNum, dropPadMode, expertTokensNumFlag, quantMode,
            rowIdxType, expertRange, xShape, expertIdxShape, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "CheckInitTilingData failed."), return ge::GRAPH_FAILED);

    // Set Param
    initTilingData.expertTokensNumType = expertTokensNumType;
    initTilingData.expertNum = expertNum;
    initTilingData.dropPadMode = dropPadMode;
    initTilingData.expertTokensNumFlag = expertTokensNumFlag ? 1 : 0;
    initTilingData.quantMode = OpQuantModeToInitRoutingQuantMode(quantMode);
    initTilingData.rowIdxType = rowIdxType;

    initTilingData.expertStart = expertRange[0];
    initTilingData.expertEnd = expertRange[1];
    initTilingData.actualExpertNum = expertRange[1] - expertRange[0];

    initTilingData.n = bs;
    initTilingData.cols = h;
    initTilingData.k = topK;
    initTilingData.isInputScale = isInputScale ? 1 : 0;
    initTilingData.isInputOffset = isInputOffset ? 1 : 0;

    initTilingData.coreNum = aivNum;

    // Reserved
    initTilingData.gatherFirstFullload = 0;
    initTilingData.activeNum = initTilingData.n * initTilingData.k;
    initTilingData.epFullload = 0;  // unsupported
    initTilingData.smoothType = 0;  // unsupported

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAuxiliaryTilingCtx(
    MoeInitRoutingV3Arch35TilingData &initTilingData, MegaMoeAuxTilingContext &ctx,
    const char* nodeName, const uint32_t aivNum, const uint64_t ubSize, ge::DataType xDtype)
{
    int64_t inputXDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_TILING_CHECK(
        inputXDtypeSize < 0,
        OP_LOGE(nodeName, "invalid x dtype(%s), expecting DT_BF16.",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    ctx.aivCoreNum = aivNum;
    ctx.totalUbSize = ubSize;
    ctx.availUbSize = ubSize - SIMT_DCACHE_SIZE;

    ctx.n = initTilingData.n;
    ctx.cols = initTilingData.cols;
    ctx.k = initTilingData.k;
    ctx.inputXDtypeSize = inputXDtypeSize;
    ctx.isInputScale = initTilingData.isInputScale;
    ctx.isInputOffset = initTilingData.isInputOffset;

    ctx.activeNum = initTilingData.activeNum;
    ctx.expertCapacity = DISABLE_EXPERT_CAPACITY;
    ctx.expertNum = initTilingData.expertNum;
    ctx.dropPadMode = initTilingData.dropPadMode;
    ctx.expertTokensNumType = initTilingData.expertTokensNumType;
    ctx.expertTokensNumFlag = initTilingData.expertTokensNumFlag;
    ctx.quantMode = initTilingData.quantMode;
    
    ctx.expertStart = initTilingData.expertStart;
    ctx.expertEnd = initTilingData.expertEnd;
    ctx.rowIdxType = initTilingData.rowIdxType;
    ctx.totalLength = initTilingData.n * initTilingData.k;

    ctx.sortLoopMaxElement = ctx.availUbSize /
        (NUM_FOUR * NUM_TWO * NUM_FOUR) /
        SORT32_ALIGN_ELEMENT *
        SORT32_ALIGN_ELEMENT;
    ctx.sortLoopMaxElement = std::min(ctx.sortLoopMaxElement, SORT_API_MAX_ELEM);

    return ge::GRAPH_SUCCESS;
}

static void Tiling4VBSOneCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
    const MegaMoeAuxTilingContext &ctx)
{
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = ctx.totalLength;
    vbsTiling->perCoreLoops = ops::CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

static void Tiling4VBSMultiCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
    MegaMoeAuxTilingContext &ctx)
{
    int64_t needCoreNum = ops::CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    needCoreNum = std::min(needCoreNum, ctx.aivCoreNum);

    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (ctx.totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = ctx.totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = ops::CeilDiv(ctx.totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = ops::CeilDiv(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = ctx.totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            ops::CeilAlign(ops::CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops), SORT32_ALIGN_ELEMENT);
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
}

static void Tiling4VBSCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
    MegaMoeAuxTilingContext &ctx)
{
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = ctx.sortLoopMaxElement;

    if (ctx.totalLength <= ctx.sortLoopMaxElement) {
        ctx.sortMode = 0;
        Tiling4VBSOneCore(vbsTiling, ctx);
    } else {
        ctx.sortMode = 1;
        Tiling4VBSMultiCore(vbsTiling, ctx);
    }
    tilingData->tilingKey = ctx.sortMode * SORT_CORE_TILINGKEY_BASE + 1031000;
}

static void Tiling4VMSMiddleCompute(MoeInitRoutingV3Arch35TilingData *tilingData)
{
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingData->vmsMiddleComputeParamsOp);

    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) {
        vmsMiddleTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = ops::CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum;
}

static void Tiling4SortOutCompute(MoeInitRoutingV3Arch35TilingData *tilingData)
{
    auto *sortOutTiling = &(tilingData->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

static void Tiling4ExpertTokensCountCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
    const MegaMoeAuxTilingContext &ctx)
{
    auto *tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = ops::CeilDiv(totalElements, ctx.aivCoreNum);
    int64_t needCoreNum = ops::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;

    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingData->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE) ?
                                   tilingData->actualExpertNum :
                                   (tilingData->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (ctx.availUbSize -
         ops::CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));

    int64_t perCoreLoops = ops::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = ops::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = ops::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = ops::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

static int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols, int64_t inputXDtypeSize, int64_t availUbSize)
{
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize) + AlignBytes(perLoopCols, sizeof(int8_t));
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    return (availUbSize - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

static void Tiling4GatherOutMxQuant(MoeInitRoutingV3Arch35TilingData *tilingData,
    const MegaMoeAuxTilingContext &ctx)
{
    auto *gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = ops::CeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = ops::CeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = ops::CeilAlign(tilingData->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = ops::CeilAlign(ops::CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    }
    int64_t colsLoops = ops::CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = ops::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = ops::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

MoeInitRoutingV3Arch35TilingData ComputeMoeInitRoutingV3Tiling(
    MoeInitRoutingV3Arch35TilingData &tilingData, MegaMoeAuxTilingContext &ctx)
{
    Tiling4VBSCompute(&tilingData, ctx);
    Tiling4VMSMiddleCompute(&tilingData);
    Tiling4SortOutCompute(&tilingData);
    Tiling4ExpertTokensCountCompute(&tilingData, ctx);
    Tiling4GatherOutMxQuant(&tilingData, ctx); // support fp8_e4m3fn, fp8_e5m2 currently
    return tilingData;
}

static ge::graphStatus GetMoeInitRoutingV3Tiling(MegaMoeTilingData &tilingData,
    const uint32_t aivNum, const uint64_t ubSize, const int64_t opQuantMode, ge::DataType xDtype, const char *nodeName)
{
    MoeInitRoutingV3Arch35TilingData initTilingData;
    OP_TILING_CHECK(
        CheckAndSetInitTilingData(tilingData, initTilingData, aivNum, opQuantMode, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "SetInitTilingData failed."), return ge::GRAPH_FAILED);

    MegaMoeAuxTilingContext ctx;
    OP_TILING_CHECK(
        SetAuxiliaryTilingCtx(initTilingData, ctx, nodeName, aivNum, ubSize, xDtype) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "SetAuxiliaryTilingCtx failed."), return ge::GRAPH_FAILED);

    printAuxiliaryTilingCtx(&ctx, tilingData.expertPerRank, tilingData.epWorldSize,
        {tilingData.bs, tilingData.topK}, xDtype, nodeName);

    auto initRoutingTilingData = ComputeMoeInitRoutingV3Tiling(initTilingData, ctx);

    int64_t leastCoreNum = std::max(
        std::max(initTilingData.vbsComputeParamsOp.needCoreNum,
            initTilingData.vmsMiddleComputeParamsOp.needCoreNum),
        std::max(initTilingData.expertTokensCountTilingDataOp.needCoreNum,
            initTilingData.gatherOutComputeParamsOp.needCoreNum));
    bool isCoreNumValid = initTilingData.coreNum >= leastCoreNum;
    OP_TILING_CHECK(
        !isCoreNumValid,
        OP_LOGE(nodeName, "aivNum(%ld) < leastCoreNum(%ld).", initTilingData.coreNum, leastCoreNum),
        return ge::GRAPH_FAILED);
    
    tilingData.moeInitRoutingTilingData = initRoutingTilingData;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto globalBs = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr,
        OP_LOGE(nodeName, "moeExpertNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(nodeName, "epWorldSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
        OP_LOGE(nodeName, "cclBufferSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr,
        OP_LOGE(nodeName, "maxRecvTokenNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr,
        OP_LOGE(nodeName, "dispatchQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutTypePtr == nullptr,
        OP_LOGE(nodeName, "dispatchQuantOutTypePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr,
        OP_LOGE(nodeName, "combineQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr,
        OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBs == nullptr,
        OP_LOGE(nodeName, "globalBs is nullptr."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);
    
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t n = weightOneStorageShape->GetStorageShape().GetDim(1);
    
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
        OP_LOGE(nodeName,
        "epWorldSize should in [%ld, %ld], but now epWorldSize is %ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
        return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
        OP_LOGE(nodeName,
            "moeExpertNum(%ld) should in [%ld, %ld] and mod(moeExpertNum, epWorldSize(%ld)) should be zero.",
            moeExpertNum, epWorldSize, MAX_MOE_EXPERT_NUM, epWorldSize),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(moeExpertNum != expertPerRank * epWorldSize,
        OP_LOGE(nodeName,
            "moeExpertNum(%ld) should be equal to expertPerRank(%ld) * epWorldSize(%ld) = %ld.",
            moeExpertNum, expertPerRank, epWorldSize, expertPerRank * epWorldSize),
        return ge::GRAPH_FAILED);

    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    int64_t cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);
    // leastCclBufferSize = PEERMEM_DATA_OFFSET + tokenPerExpert + quantTokenScale + combineOut
    int64_t leastCclBufferSize = PEERMEM_DATA_OFFSET +
        (epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, ALIGN_128) * sizeof(int32_t)) + // tokenPerExpert
        (ops::CeilAlign(bs * topK * (h + h / MXFP_SCALE_GROUP_NUM), ALIGN_512) * sizeof(int8_t)) + // quantTokenScale
        (ops::CeilAlign(bs * h * topK * yDtypeSize, ALIGN_512)); // combineOut
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
        OP_LOGE(nodeName, "cclBufferSize(%ld) should equal or larger than leastCclBufferSize(%ld).",
            cclBufferSize, leastCclBufferSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    int64_t maxRecvTokenNum = static_cast<int64_t>(*maxRecvTokenNumPtr);
    OP_TILING_CHECK(maxRecvTokenNum < 0 || maxRecvTokenNum > bs * epWorldSize * std::min(topK, expertPerRank),
        OP_LOGE(nodeName,
            "maxRecvTokenNum(%ld) should in [0, %ld], right bound is bs * epWorldSize * min(topK, expertPerRank).",
            maxRecvTokenNum, bs * epWorldSize * std::min(topK, expertPerRank)),
        return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP,
        OP_LOGE(nodeName,
            "Invalid dispatchQuantMode, only support mxfp(%ld), but now is %ld.",
            DISPATCH_QUANT_MODE_MXFP, dispatchQuantMode),
        return ge::GRAPH_FAILED);

    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);
    OP_TILING_CHECK(dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) &&
                    dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
                    dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "Invalid dispatchQuantOutType, only support fp8_e5m2, fp8_e4m3fn and fp4_e2m1, "
            "but now is %ld.", dispatchQuantOutType),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName,
            "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.", dispatchQuantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((refWeightDataType != weightOneDesc->GetDataType()) &&
                    (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1),
        OP_LOGE(nodeName, "refWeightDataType(%s) should be equal to weightOne & weightTwo dataType(%s).",
        ge::TypeUtils::DataTypeToSerialString(refWeightDataType).c_str(),
        ge::TypeUtils::DataTypeToSerialString(weightOneDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(*combineQuantModePtr != 0,
        OP_LOGE(nodeName, "Invalid combineQuantMode(%ld), expecting default value(0).", *combineQuantModePtr),
        return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
        OP_LOGE(nodeName, "Invalid commAlg(%s), expecting default value("").", commAlgPtr),
        return ge::GRAPH_FAILED);

    auto globalBsPtr = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));
    int64_t globalBs = static_cast<int64_t>(*globalBsPtr);
    if (globalBs != 0) {
        OP_TILING_CHECK(globalBs < 0 || bs * epWorldSize > globalBs || globalBs % epWorldSize != 0,
            OP_LOGE(nodeName, "globalBs(%ld) either be 0 or (maxBs * EP and mod(globalBs, EP(%ld)) == 0).",
                globalBs, epWorldSize),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));

    tilingData->epWorldSize = *epWorldSizePtr;
    tilingData->maxOutputSize = *maxRecvTokenNumPtr != 0 ?
        *maxRecvTokenNumPtr :
        tilingData->bs * tilingData->epWorldSize *
        std::min(tilingData->topK, tilingData->expertPerRank);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->dispatchRows = 2;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(
    gert::TilingContext *context, WorkspaceInfo& workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace is nullptr."),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(workspaceInfo.workspaceSize == 0LL,
        OP_LOGE(nodeName, "workspaceSize from tilingData is 0."),
        return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr,
        OP_LOGE(nodeName, "xActiveMaskDesc should be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE(nodeName, "scalesDesc should be null."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_TILING_CHECK(weightOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight1",
        (std::to_string(weightOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    const int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightOne dim0 = %ld", weightOneDim0);
    OP_LOGD(nodeName, "weightOne dim1 = %ld", weightOneDim1);
    OP_LOGD(nodeName, "weightOne dim2 = %ld", weightOneDim2);

    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weightTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    OP_TILING_CHECK(weightTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight2",
        (std::to_string(weightTwoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    const int64_t weightTwoDim0 = weightTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightTwoDim1 = weightTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightTwoDim2 = weightTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightTwo dim0 = %ld", weightTwoDim0);
    OP_LOGD(nodeName, "weightTwo dim1 = %ld", weightTwoDim1);
    OP_LOGD(nodeName, "weightTwo dim2 = %ld", weightTwoDim2);

    OP_TILING_CHECK(weightOneDim0 != weightTwoDim0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1 and weight2",
        (std::string("[") + std::to_string(weightOneDim0) + ", " + std::to_string(weightTwoDim0) + "]").c_str(),
        "dim0 of weight1 and weight2 should be equal"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim2 != weightTwoDim1 || h != weightOneDim2,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1, weight2 and x",
        (std::string("[") + std::to_string(weightOneDim2) + ", " + std::to_string(weightTwoDim1) +
         ", " + std::to_string(h) + "]").c_str(),
        "dim2 of weight1 and dim1 of weight2 should equal h of x"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDim1 != weightTwoDim2 * NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1 and weight2",
        (std::string("[") + std::to_string(weightOneDim1) + ", " + std::to_string(weightTwoDim2) + "]").c_str(),
        "dim1 of weight1 should equal dim2 of weight2 * 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);

    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "y",
        (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(yDim0 != bs || yDim1 != h,
        OP_LOGE(nodeName, "(yDim0(%ld), yDim1(%ld)) should be equal to (bs(%ld), h(%ld)).",
            yDim0, yDim1, bs, h),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_token_nums",
        (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    OP_TILING_CHECK(expertTokenNumsDim0 != expertPerRank,
        OP_LOGE(nodeName, "expertTokenNumsDim0(%ld) should be equal to expertPerRank(%ld).",
            expertTokenNumsDim0, expertPerRank),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScalesOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    OP_TILING_CHECK(weightScalesOneStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight_scales1",
        (std::to_string(weightScalesOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
        return ge::GRAPH_FAILED);
    const int64_t weightScalesOneDim0 = weightScalesOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesOneDim1 = weightScalesOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesOneDim2 = weightScalesOneStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesOneDim3 = weightScalesOneStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesOne dim0 = %ld", weightScalesOneDim0);
    OP_LOGD(nodeName, "weightScalesOne dim1 = %ld", weightScalesOneDim1);
    OP_LOGD(nodeName, "weightScalesOne dim2 = %ld", weightScalesOneDim2);
    OP_LOGD(nodeName, "weightScalesOne dim3 = %ld", weightScalesOneDim3);

    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScalesTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);
    OP_TILING_CHECK(weightScalesTwoStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE(nodeName, "weightScalesTwoStorageShape dims must be 4, but current dim num is %zu.",
        weightScalesTwoStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t weightScalesTwoDim0 = weightScalesTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesTwoDim1 = weightScalesTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesTwoDim2 = weightScalesTwoStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesTwoDim3 = weightScalesTwoStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesTwo dim0 = %ld", weightScalesTwoDim0);
    OP_LOGD(nodeName, "weightScalesTwo dim1 = %ld", weightScalesTwoDim1);
    OP_LOGD(nodeName, "weightScalesTwo dim2 = %ld", weightScalesTwoDim2);
    OP_LOGD(nodeName, "weightScalesTwo dim3 = %ld", weightScalesTwoDim3);

    OP_TILING_CHECK(weightScalesOneDim0 != weightScalesTwoDim0,
        OP_LOGE(nodeName, "weightScalesOneDim0(%ld) and weightScalesTwoDim0(%ld) should be equal.",
            weightScalesOneDim0, weightScalesTwoDim0),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDim3 != NUM_TWO || weightScalesOneDim3 != weightScalesTwoDim3,
        OP_LOGE(nodeName, "weightScalesOneDim3(%ld) and weightScalesTwoDim3(%ld) should be equal to 2.",
            weightScalesOneDim3, weightScalesTwoDim3),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightScalesTwoDim1 != h,
        OP_LOGE(nodeName, "weightScalesTwoDim1(%ld) should be equal to h(%ld).",
            weightScalesTwoDim1, h),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScalesOneDim2 != ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
            "weightScalesOneDim2(%ld) should equal CeilDiv(h(%ld), INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
            weightScalesOneDim2, h,
            INPUT_WEIGHT_SCALES_CEIL_ALIGN, ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    const int64_t n = weightScalesOneDim1;
    OP_TILING_CHECK(weightScalesTwoDim2 != ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
            "weightScalesTwoDim2(%ld) should equal CeilDiv(n(%ld) / 2, INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
            weightScalesTwoDim2, n,
            INPUT_WEIGHT_SCALES_CEIL_ALIGN, ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr,
        OP_LOGE(nodeName, "contextShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %zu.",
        contextStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %zu.",
        xStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr,
        OP_LOGE(nodeName, "topkIdsStorageShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkIdsStorageShape dims must be 2, but current dim num is %zu.",
        topkIdsStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr,
        OP_LOGE(nodeName, "topkWeightsStorageShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkWeightsStorageShape dims must be 2, but current dim num is %zu.",
        topkWeightsStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 && xDim0 != topkWeightsDim0 && topkIdsDim0 != topkWeightsDim0,
        OP_LOGE(nodeName, "xDim0(%ld), topkIdsDim0(%ld), topkWeightsDim0(%ld) should all be equal.",
            xDim0, topkIdsDim0, topkWeightsDim0),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDim1 != topkWeightsDim1,
        OP_LOGE(nodeName, "topkIdsDim1(%ld), topkWeightsDim1(%ld) should be equal.",
            topkIdsDim1, topkWeightsDim1),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "optional params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    
    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "context dataType is invalid, dataType should be DT_INT32, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(contextDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be DT_BF16, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "topkIds dataType is invalid, dataType should be DT_INT32, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(topkIdsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((
        (topkWeightsDesc->GetDataType() != ge::DT_BF16) &&
        (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be DT_FLOAT or DT_BF16, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(topkWeightsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((
        (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
        (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
        (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "weightOne dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1, "
            "but is %s.", ge::TypeUtils::DataTypeToSerialString(weightOneDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK((
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "weightTwo dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1, "
            "but is %s.", ge::TypeUtils::DataTypeToSerialString(weightTwoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE(nodeName,
            "weightOne dataType(%s) and weightTwo dataType(%s) should be equal.",
        ge::TypeUtils::DataTypeToSerialString(weightOneDesc->GetDataType()).c_str(),
        ge::TypeUtils::DataTypeToSerialString(weightTwoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName,
            "weightScalesOne dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(weightScalesOneDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName,
            "weightScalesTwo dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(weightScalesTwoDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE(nodeName, "y dataType is invalid, dataType should be DT_BF16, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be DT_INT32, but is %s.",
        ge::TypeUtils::DataTypeToSerialString(expertTokenNumsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    
    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "topkIds format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "topkWeights format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightOne format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightTwoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightTwo format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "y format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "expertTokenNums format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(xDim0 < MIN_BS || xDim0 > MAX_BS,
        OP_LOGE(nodeName, "BS should in [%ld, %ld], but now BS is %ld.", MIN_BS, MAX_BS, xDim0),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim1 != 4LL * H_BASE && xDim1 != 5LL * H_BASE && xDim1 != 7LL * H_BASE,
        OP_LOGE(nodeName, "H only support 4k/5k/7k, but now H is %ld.", xDim1),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(topkIdsDim1 != 6LL && topkIdsDim1 != 8LL,
        OP_LOGE(nodeName, "topK only support 6/8, but now topK is %ld.", topkIdsDim1),
        return ge::GRAPH_FAILED);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightOneDim0 < MIN_EXPERT_PER_RANK || weightOneDim0 > MAX_EXPERT_PER_RANK,
        OP_LOGE(nodeName,
            "expertPerRank should in [%ld, %ld], but now expertPerRank is %ld.",
            MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, weightOneDim0),
        return ge::GRAPH_FAILED);

    int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim1 != 1LL * HIDDEN_DIM_BASE && weightOneDim1 != 2LL * HIDDEN_DIM_BASE &&
                    weightOneDim1 != 3LL * HIDDEN_DIM_BASE && weightOneDim1 != 4LL * HIDDEN_DIM_BASE &&
                    weightOneDim1 != 7LL * HIDDEN_DIM_BASE,
        OP_LOGE(nodeName, "hiddenDim only support 1k/2k/3k/4k/7k, but now hiddenDim is %ld.", weightOneDim1),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context,
    MegaMoeTilingData *tilingData, MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenDim = weightOneStorageShape->GetStorageShape().GetDim(1);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(h);
    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->expertPerRank = static_cast<uint32_t>(expertPerRank);
    tilingData->groupListType = 1;

    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
    MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
        OP_LOGE(nodeName, "aivNum(%u) or aicNum(%u) less than or equal 0.", aivNum, aicNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // InitRouting
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType xDtype = context->GetInputDesc(config.xIndex)->GetDataType();
    OP_TILING_CHECK(
        GetMoeInitRoutingV3Tiling(*tilingData, aivNum, ubSize, opQuantMode, xDtype, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Moe init routing failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t*)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    printWorkspaceInfo(&workspaceInfo, nodeName);
    printPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext* context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe)
    .Tiling(MegaMoeTilingFunc)
    .TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);
} // namespace optiling
