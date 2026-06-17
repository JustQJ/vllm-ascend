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
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif

namespace Blaze {
namespace Gemm {
constexpr int32_t MATMUL_MNK_ALIGN = 16;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2LL;
constexpr int64_t QUADRUPLE_BUFFER_COUNT = 4LL;
constexpr int32_t MNK_M = 0;
constexpr int32_t MNK_N = 1;
constexpr int32_t MNK_K = 2;
constexpr int32_t MNK_B = 3;
constexpr int32_t MNK_M0 = 4;
constexpr int32_t MNK_N0 = 5;
constexpr int32_t EVEN_NUMBER = 2;

constexpr static uint64_t A_FULL_LOAD_MODE = 1UL;
constexpr static uint64_t B_FULL_LOAD_MODE = 2UL;
constexpr static uint64_t NONE_FULL_LOAD_MODE = 0UL;

constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint16_t SECOND_FLAG = 2;
constexpr uint16_t THIRD_FLAG = 3;
constexpr uint16_t FOURTH_FLAG = 4;
constexpr uint16_t FIFTH_FLAG = 5;
constexpr uint16_t SIXTH_FLAG = 6;
constexpr uint16_t SEVENTH_FLAG = 7;
constexpr static int64_t PER_BLOCK_SIZE = 128LL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BLOCK_CUBE = 16UL;
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

constexpr uint64_t C0_SIZE_fp16 = 16UL;
constexpr uint64_t C0_SIZE_fp32 = 8UL;
constexpr uint64_t C0_SIZE_B8 = 32UL;
constexpr uint64_t C0_SIZE_B4 = 64UL;
constexpr uint64_t C0_SIZE_L0C = 16UL;

// FusedMatMul OpType
constexpr uint64_t OP_TYPE_EMPTY = 0UL;
constexpr uint64_t OP_TYPE_ADD = 1UL;
constexpr uint64_t OP_TYPE_MUL = 2UL;
constexpr uint64_t OP_TYPE_RELU = 5UL;
constexpr int64_t BLOCK_BYTE_SIZE = 32L;

constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;

constexpr uint64_t IDX_M_IDX = 0UL;
constexpr uint64_t IDX_N_IDX = 1UL;
constexpr uint64_t IDX_K_IDX = 2UL;

constexpr uint16_t INPUT_BUFFER_FLAG_0 = 0;
constexpr uint16_t INPUT_BUFFER_FLAG_1 = 1;
constexpr uint16_t INPUT_BUFFER_FLAG_2 = 2;
constexpr uint16_t INPUT_BUFFER_FLAG_3 = 3;

enum class QuantMode : uint32_t
{
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

template <typename...>
struct always_false : public AscendC::Std::false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
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

__aicore__ inline int64_t GetPerBlockNum(int64_t coreNum, int64_t mTileNum, int64_t nTileNum, int64_t b = 1)
{
    int64_t perCoreBlockNum = Blaze::Gemm::CeilDiv(mTileNum * nTileNum * b, coreNum);
    return perCoreBlockNum;
}

__aicore__ inline uint64_t CalWeightNZGmAddrOffset(bool transB, int64_t batchIdx, int64_t n, int64_t k, int64_t c0_size)
{
    if (transB) {
        return batchIdx * Blaze::Gemm::CeilDiv(k, c0_size) * Blaze::Gemm::CeilDiv(n, static_cast<int64_t>(BLOCK_CUBE)) *
               BLOCK_CUBE * c0_size;
    } else {
        return batchIdx * Blaze::Gemm::CeilDiv(n, c0_size) * Blaze::Gemm::CeilDiv(k, static_cast<int64_t>(BLOCK_CUBE)) *
               BLOCK_CUBE * c0_size;
    }
}

} // namespace Gemm
} // namespace Blaze
