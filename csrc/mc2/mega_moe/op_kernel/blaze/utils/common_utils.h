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
 * \file common_utils.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Blaze {
namespace Gemm {
constexpr int32_t MATMUL_MNK_ALIGN = 16;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr int MNK_M = 0;
constexpr int MNK_N = 1;
constexpr int MNK_K = 2;
constexpr int MNK_B = 3;
constexpr int MNK_M0 = 4;
constexpr int MNK_N0 = 5;

constexpr static uint64_t A_FULL_LOAD_MODE = 1UL;
constexpr static uint64_t B_FULL_LOAD_MODE = 2UL;
constexpr static uint64_t NONE_FULL_LOAD_MODE = 0UL;

constexpr static int64_t PER_BLOCK_SIZE = 128LL;
constexpr uint32_t MXFP_DIVISOR_SIZE = 64;
constexpr uint32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint32_t BLOCK_CUBE = 16UL;
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr uint32_t C0_SIZE_B8 = 32UL;
constexpr uint32_t C0_SIZE_B4 = 64UL;
constexpr uint32_t C0_SIZE_L0C = 16UL;

struct MatmulShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t b;
};

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

} // namespace Gemm
} // namespace Blaze
