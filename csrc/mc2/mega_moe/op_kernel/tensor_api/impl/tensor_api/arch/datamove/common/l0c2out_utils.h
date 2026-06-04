/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/datamove/common/l0c2out_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_L0C2OUT_UTILS_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_L0C2OUT_UTILS_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/datamove/common/instruction.h"

namespace AscendC {
namespace Te {

constexpr uint32_t MAIN_LOOP_N_SIZE_3510 = 512;
constexpr uint32_t CBURST_NUM_3510 = MAIN_LOOP_N_SIZE_3510 / BLOCK_CUBE;

constexpr FixpipeParams DEFAULT_FIXPIPE_PARAMS = FixpipeParams{};

template <RoundMode roundMode, typename dstType, typename srcType>
__aicore__ inline constexpr QuantMode_t GetVectorQuantMode()
{
    if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, half>) {
        return QuantMode_t::VDEQF16;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, uint8_t, int8_t>) {
        return QuantMode_t::VQF322B8_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, uint8_t, int8_t>) {
        return QuantMode_t::VREQ8;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, fp8_e4m3fn_t>) {
        return QuantMode_t::VQF322FP8_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, hifloat8_t>) {
        if constexpr (roundMode == RoundMode::HYBRID) {
            return QuantMode_t::VQF322HIF8_PRE_HYBRID;
        } else {
            return QuantMode_t::VQF322HIF8_PRE;
        }
    } else if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, bfloat16_t>) {
        return QuantMode_t::VQS322BF16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, half>) {
        return QuantMode_t::VQF322F16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, bfloat16_t>) {
        return QuantMode_t::VQF322BF16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, float>) {
        return QuantMode_t::VQF322F32_PRE;
    }
}

template <RoundMode roundMode, typename dstType, typename srcType>
__aicore__ inline constexpr QuantMode_t GetScalarQuantMode()
{
    if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, half>) {
        return QuantMode_t::DEQF16;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, uint8_t, int8_t>) {
        return QuantMode_t::QF322B8_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, uint8_t, int8_t>) {
        return QuantMode_t::REQ8;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, fp8_e4m3fn_t>) {
        return QuantMode_t::QF322FP8_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, hifloat8_t>) {
        if constexpr (roundMode == RoundMode::HYBRID) {
            return QuantMode_t::QF322HIF8_PRE_HYBRID;
        } else {
            return QuantMode_t::QF322HIF8_PRE;
        }
    } else if constexpr (IsOneOfAttrV<srcType, int32_t> && IsOneOfAttrV<dstType, bfloat16_t>) {
        return QuantMode_t::QS322BF16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, half>) {
        return QuantMode_t::QF322F16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, bfloat16_t>) {
        return QuantMode_t::QF322BF16_PRE;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, float>) {
        return QuantMode_t::QF322F32_PRE;
    }
}

template <RoundMode roundMode, typename dstType, typename srcType>
__aicore__ inline constexpr QuantMode_t GetCastQuantMode()
{
    if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, half>) {
        return QuantMode_t::F322F16;
    } else if constexpr (IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, bfloat16_t>) {
        return QuantMode_t::F322BF16;
    } else {
        return QuantMode_t::NoQuant;
    }
}

template <RoundMode roundMode, typename T, typename U, typename S = void>
__aicore__ inline constexpr QuantMode_t GetQuantMode()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    constexpr bool isTensor = IsAttrTensorV<S>;
    constexpr bool isScalar = Std::is_same_v<S, uint64_t>;

    if constexpr (roundMode == RoundMode::HYBRID) {
        static_assert((IsOneOfAttrV<srcType, float> && IsOneOfAttrV<dstType, hifloat8_t>),
                      "Only when L0CType is float and output Type is hifloat8_t support RoundMode::HYBRID in Fixpipe");
    }
    if constexpr (isTensor) {
        return GetVectorQuantMode<roundMode, dstType, srcType>();
    } else if constexpr (isScalar) {
        return GetScalarQuantMode<roundMode, dstType, srcType>();
    } else {
        return GetCastQuantMode<roundMode, dstType, srcType>();
    }
}

template <typename T, typename U>
__aicore__ inline static void SetRegisterImpl(const T& /*dst*/, const U& /*src*/)
{
    if constexpr (IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) {
        constexpr uint32_t ndNum = 1;
        constexpr uint32_t srcNdStride = 0;
        constexpr uint32_t dstNdStride = 0;
        SetRegisterInstr::SetRegister(ndNum, dstNdStride, srcNdStride);
    } else if constexpr (IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>) {
        constexpr uint32_t dnNum = 1;
        constexpr uint32_t dstDnMatrixStride = 0;
        constexpr uint32_t srcNzMatrixStride = 0;
        constexpr uint32_t srcNzC0Stride = 1;
        SetRegisterInstr::SetRegister(dnNum, dstDnMatrixStride, srcNzMatrixStride, srcNzC0Stride);
    }
}

template <typename T, typename U>
__aicore__ inline static void SetRegisterImpl(const T& /*dst*/, const U& /*src*/, uint64_t quant)
{
    if constexpr (IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) {
        constexpr uint32_t ndNum = 1;
        constexpr uint32_t srcNdStride = 0;
        constexpr uint32_t dstNdStride = 0;
        SetRegisterInstr::SetRegister(quant, ndNum, dstNdStride, srcNdStride);
    } else if constexpr (IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>) {
        constexpr uint32_t dnNum = 1;
        constexpr uint32_t dstDnMatrixStride = 0;
        constexpr uint32_t srcNzMatrixStride = 0;
        constexpr uint32_t srcNzC0Stride = 1;
        SetRegisterInstr::SetRegister(quant, dnNum, dstDnMatrixStride, srcNzMatrixStride, srcNzC0Stride);
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_L0C2OUT_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif