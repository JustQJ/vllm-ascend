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
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

namespace Blaze {
namespace Gemm {
/* block schedule policies */
struct KernelMmadWithScaleMx {};   // Multi-block with Mx scale
struct KernelMmadWithScaleFixpipeQuant {}; // Multi-block with fixpipe quant scale (A8W8 fixpipe)
struct KernelMultiBlockStreamK {}; // Multi-tile transfer with K-axis spliting and caching
struct KernelMmadMultiBlockBasic {}; // Multi-tile basic
struct KernelMmadMultiBlockBmmBroadcast {}; // Multi-tile batchMatmul broadcast
enum class MatMulL0C2Out : std::uint8_t {
    ON_THE_FLY = 0,
    ND_FIXPIPE_1_1 = 1,
    ND_FIXPIPE_1_2 = 2
};

/**
 * @struct MatmulWithScaleFixpipeQuant
 * @brief Quantized fixpipe matmul with scale and fixpipe dequant (Tensor API / Blaze)
 * @param [in] FULL_LOAD_MODE_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] ATOMIC_ADD: whether to enable atomic add on output
 */
template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD_ = false>
struct MatmulWithScaleFixpipeQuant {
    using ScheduleType = KernelMmadWithScaleFixpipeQuant;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD_;
};

/**
 * @struct MatmulWithScaleMx
 * @brief Mx Matrix multiplication with scaleA and scaleB
 */
template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD = false>
struct MatmulWithScaleMx {
    using ScheduleType = KernelMmadWithScaleMx;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] FUSED_OP_TYPE_: execute fusion after mmad , default is 0
 */
template <MatMulL0C2Out FixpOpti = MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE_ = 0>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelMultiBlockStreamK;
    constexpr static bool enableInputDataLenCheck = false;
    constexpr static bool enableRelu = (FUSED_OP_TYPE_ == OP_TYPE_RELU);
    constexpr static bool enableAdd = (FUSED_OP_TYPE_ == OP_TYPE_ADD);
    constexpr static bool enableMul = (FUSED_OP_TYPE_ == OP_TYPE_MUL);
    constexpr static MatMulL0C2Out fixpOpti_ = FixpOpti;
};

/**
 * @struct MatmulMultiBlockBasic
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FULL_LOAD_MODE: mode of full load, default is 0(no full load)
 * @param [in] FUSED_OP_TYPE_: execute fusion after mmad , default is 0
 */
template <uint64_t FULL_LOAD_MODE_ = 0, uint64_t FUSED_OP_TYPE_ = 0, class KernelSchedule_ = KernelMmadMultiBlockBasic>
struct MatmulMultiBlockBasic {
    using ScheduleType = KernelSchedule_;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool enableRelu = (FUSED_OP_TYPE_ == OP_TYPE_RELU);
    constexpr static bool enableAdd = (FUSED_OP_TYPE_ == OP_TYPE_ADD);
    constexpr static bool enableMul = (FUSED_OP_TYPE_ == OP_TYPE_MUL);
};

} // namespace Gemm
} // namespace Blaze
