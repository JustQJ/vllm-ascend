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
    "impl/tensor_api/arch/cube/utils/l0c2out_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file l0c2out_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace AscendC {
namespace Te {

constexpr uint32_t MAIN_LOOP_N_SIZE = 512;
constexpr uint32_t CBURST_NUM = MAIN_LOOP_N_SIZE / BLOCK_CUBE;

constexpr FixpipeParams DEFAULT_FIXPIPE_PARAMS = FixpipeParams{};

template <typename TensorType>
inline constexpr bool IsL0COutSrcBatchLayoutV = TensorType::layoutType::depth == FIVE_DIM_DATA;

template <typename TensorType>
inline constexpr bool IsL0COutNDFormatV =
    IsSatisfiedPtnFormatV<TensorType, NDExtLayoutPtn> || IsSatisfiedPtnFormatV<TensorType, NDLayoutPtn>;

template <typename TensorType>
inline constexpr bool IsL0COutDNFormatV =
    IsSatisfiedPtnFormatV<TensorType, DNExtLayoutPtn> || IsSatisfiedPtnFormatV<TensorType, DNLayoutPtn>;

template <typename T, typename LayoutType>
__aicore__ inline static constexpr uint32_t GetL0COutNDStride(const LayoutType& layout)
{
    if constexpr (IsSatisfiedPtnFormatV<T, NDLayoutPtn>) {
        return GetElement<AttrInfo::Stride, AttrInfo::Row>(layout);
    } else {
        return GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(layout);
    }
}

template <typename T, typename LayoutType>
__aicore__ inline static constexpr uint32_t GetL0COutDNStride(const LayoutType& layout)
{
    if constexpr (IsSatisfiedPtnFormatV<T, DNLayoutPtn>) {
        return GetElement<AttrInfo::Stride, AttrInfo::Column>(layout);
    } else {
        return GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(layout);
    }
}

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


class SetRegisterInstr {
public:
    __aicore__ inline static void SetRegister(uint64_t quant, uint32_t ndNum, uint32_t dstNDStride,
                                              uint32_t srcNDStride)
    {
        SetQuantPre(quant);
        SetLoop3Para<uint64_t>(ndNum, dstNDStride, srcNDStride);
    }

    __aicore__ inline static void SetRegister(uint64_t quant, uint32_t dnNum, uint32_t dstDNStride,
                                              uint32_t srcNZMatrixStride, uint32_t srcNZC0Stride)
    {
        SetQuantPre(quant);
        SetLoop3Para<uint64_t>(dnNum, dstDNStride, srcNZMatrixStride);
        SetChannelPara<uint64_t>(srcNZC0Stride);
    }

    __aicore__ inline static void SetRegister(uint32_t ndNum, uint32_t dstNDStride, uint32_t srcNDStride)
    { SetLoop3Para<uint64_t>(ndNum, dstNDStride, srcNDStride); }

    __aicore__ inline static void SetRegister(uint32_t dnNum, uint32_t dstDNStride, uint32_t srcNZMatrixStride,
                                              uint32_t srcNZC0Stride)
    {
        SetLoop3Para<uint64_t>(dnNum, dstDNStride, srcNZMatrixStride);
        SetChannelPara<uint64_t>(srcNZC0Stride);
    }

private:
    static constexpr uint32_t SHIFT_LOOP3_DST_STRIDE = 32;
    static constexpr uint32_t SHIFT_LOOP3_SRC_MATRIX = 16;
    static constexpr uint32_t SHIFT_CHANNEL_C0_STRIDE = 48;

    __aicore__ inline static void SetQuantPre(uint64_t quant)
    {
        if ASCEND_IS_AIV {
            return;
        }
        
        asc_set_l0c_copy_prequant(quant);
    }

    template <typename T>
    __aicore__ inline static void SetLoop3Para(uint32_t num, uint32_t dstStride, uint32_t srcStride)
    {
        asc_set_l0c2gm_nz2nd(static_cast<T>(num), static_cast<T>(srcStride), static_cast<T>(dstStride));
    }

    template <typename T>
    __aicore__ inline static void SetChannelPara(uint32_t srcNZC0Stride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        T channelPara = 0;
        channelPara |= static_cast<T>(srcNZC0Stride) << SHIFT_CHANNEL_C0_STRIDE;
        asc_set_l0c2gm_channel_para(channelPara);
    }
};

__aicore__ inline auto AllocFbTempBuf(const uint16_t& /* calNSize */)
{
    if ASCEND_IS_AIV {
        return 0UL;
    }
    uint64_t deqTensorTempBuf = 0;
    deqTensorTempBuf = reinterpret_cast<uint64_t>(asc_get_phy_buf_addr(0));
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void SetFpc(const __fbuf__ T* deqTensorTempBuf)
{
    if ASCEND_IS_AIV {
        return;
    }
   
    uint64_t deqTensorAddr = (reinterpret_cast<uint64_t>(deqTensorTempBuf) >> 7) << 8;
    asc_set_l0c_copy_prequant(deqTensorAddr);
}

__aicore__ inline void InsertSync()
{
    if ASCEND_IS_AIV {
        return;
    }
   
    asc_sync_pipe(PIPE_FIX);
}


template <typename T, typename U>
__aicore__ inline static void EmitSetRegister(const U& srcLayout, uint32_t batchNum, uint32_t dstBatchStride,
                                              uint32_t srcBatchStride)
{
    if constexpr (IsL0COutNDFormatV<T>) {
        SetRegisterInstr::SetRegister(batchNum, dstBatchStride, srcBatchStride);
    } else if constexpr (IsL0COutDNFormatV<T>) {
        SetRegisterInstr::SetRegister(batchNum, dstBatchStride, srcBatchStride,
                                      GetElement<AttrInfo::Stride, AttrInfo::Column, 0>(srcLayout));
    }
}

template <typename T, typename U>
__aicore__ inline static void EmitSetRegister(
    const U& srcLayout, uint64_t quant, uint32_t batchNum, uint32_t dstBatchStride, uint32_t srcBatchStride)
{
    if constexpr (IsL0COutNDFormatV<T>) {
        SetRegisterInstr::SetRegister(quant, batchNum, dstBatchStride, srcBatchStride);
    } else if constexpr (IsL0COutDNFormatV<T>) {
        SetRegisterInstr::SetRegister(quant, batchNum, dstBatchStride, srcBatchStride,
                                      GetElement<AttrInfo::Stride, AttrInfo::Column, 0>(srcLayout));
    }
}

template <typename T, typename U>
__aicore__ inline static void SetRegisterImpl(const T& dst, const U& src)
{
    if constexpr (IsL0COutSrcBatchLayoutV<U>) {
        auto srcLayout = src.Layout();
        auto dstLayout = dst.Layout();
        EmitSetRegister<T>(Te::Get<1>(srcLayout), Get<0>(srcLayout.Shape()), Get<0>(dstLayout.Stride()),
                           Get<0>(srcLayout.Stride()) / FRACTAL_FIXED);
    } else {
        EmitSetRegister<T>(src.Layout(), 1, 0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline static void SetRegisterImpl(const T& dst, const U& src, uint64_t quant)
{
    if constexpr (IsL0COutSrcBatchLayoutV<U>) {
        auto srcLayout = src.Layout();
        auto dstLayout = dst.Layout();
        EmitSetRegister<T>(Te::Get<1>(srcLayout), quant, Get<0>(srcLayout.Shape()), Get<0>(dstLayout.Stride()),
                           Get<0>(srcLayout.Stride()) / FRACTAL_FIXED);
    } else {
        EmitSetRegister<T>(src.Layout(), quant, 1, 0, 0);
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
