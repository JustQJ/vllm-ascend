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
 * \file quant_batch_matmul_constant.h
 * \brief
 */
#pragma once
namespace Blaze {
namespace Gemm {
namespace QuantBatchMatmul {

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
} // namespace QuantBatchMatmul
} // namespace Gemm
} // namespace Blaze
