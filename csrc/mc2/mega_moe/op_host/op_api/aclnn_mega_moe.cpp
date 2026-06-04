/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "aclnn_mega_moe.h"
#include "opdev/op_log.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

using namespace op;

#define MEGA_MOE_CHECK_NULL(ptr, ret) \
    do {                              \
        if ((ptr) == nullptr) {       \
            return ret;               \
        }                             \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMegaMoeGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* topkIds, const aclTensor* topkWeights,
    const aclTensorList* weight1, const aclTensorList* weight2, const aclTensorList* weightScales1Optional,
    const aclTensorList* weightScales2Optional, const aclTensor* xActiveMaskOptional,
    const aclTensor* scalesOptional, int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType, int64_t combineQuantMode,
    const char* commAlg, int64_t globalBs, aclTensor* yOut, aclTensor* expertTokenNumsOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerMegaMoe(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* topkIds, const aclTensor* topkWeights,
    const aclTensorList* weight1, const aclTensorList* weight2, const aclTensorList* weightScales1Optional,
    const aclTensorList* weightScales2Optional, const aclTensor* xActiveMaskOptional,
    const aclTensor* scalesOptional, int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutType, int64_t combineQuantMode,
    const char* commAlg, int64_t globalBs, aclTensor* yOut, aclTensor* expertTokenNumsOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_LOGD("aclnn_mega_moe WorkspaceSize start");

    MEGA_MOE_CHECK_NULL(context, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(x, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(topkIds, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(topkWeights, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(weight1, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(weight2, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(yOut, ACLNN_ERR_PARAM_NULLPTR);
    MEGA_MOE_CHECK_NULL(expertTokenNumsOut, ACLNN_ERR_PARAM_NULLPTR);

    return aclnnInnerMegaMoeGetWorkspaceSize(
        context, x, topkIds, topkWeights, weight1, weight2,
        weightScales1Optional, weightScales2Optional, xActiveMaskOptional, scalesOptional,
        moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode, dispatchQuantOutType,
        combineQuantMode, commAlg, globalBs, yOut, expertTokenNumsOut, workspaceSize, executor);
}

aclnnStatus aclnnMegaMoe(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_mega_moe start");
    return aclnnInnerMegaMoe(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
