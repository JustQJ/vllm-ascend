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
 * \file moe_v3_common.h
 * \brief
 */
#ifndef MOE_V3_COMMON_H_REGBASE
#define MOE_V3_COMMON_H_REGBASE

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
#define OVERFLOW_MODE_CTRL 60
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t GATHER = 0;
constexpr int64_t SCATTER = 1;

constexpr uint16_t FLOAT_REG_TENSOR_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

// 异常值：负0
constexpr int32_t NEG_ZERO = 0x80000000;
// fp32中指数部分Mask，同时也表示fp32的INF值
constexpr uint32_t FP32_EMASK_AND_INF_VAL = 0x7f800000;
// fp32指数位的偏移位数
constexpr int16_t FP32_EXP_SHR_BITS = 23;
// fp32指数位偏移
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
// 对于x的目标类型为e2m1的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E2M1 = 0x0100;
constexpr int64_t FP4_NUMBER = 2U;

// MX量化一个块的元素数量为32个，即1个scale对应32个x的元素
constexpr int64_t MX_BLOCK_SIZE = 32LL;
// 一个VL放多少fp32->fp8的元素个数，即按fp32算的一个VL能放几个元素
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
// fp16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
// bf16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
// bfloat16的nan值（与inf值不同）
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
// 对于x的目标类型为e5m2的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
// 对于x的目标类型为e4m3的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
// e8m0的inf/nan值（按定义应该是nan值）
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
// e8m0的极小值，用于写0
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
// bf16指数位的偏移位数
constexpr int16_t BF16_EXP_SHR_BITS = 7;
// 用于计算halfScale=BF16_EXP_INVSUB-sharedExp，得到的halfScale就是1/realScale，可以用于量化xQuant=x*halfScale
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}
template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COMMON_H_REGBASE