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
    "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scaleb_dn2nn.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file scaleb_dn2nn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_NPU_ARCH_3510_GM_TO_L1_SCALEB_DN2NN_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_NPU_ARCH_3510_GM_TO_L1_SCALEB_DN2NN_H

#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufScaleBDN2Nn {
public:
    template <const CopyGM2L1Trait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        DataCopyImpl<trait>(dst, src);
    }

private:
    template <const CopyGM2L1Trait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckLayoutPattern<U, T>();
        CheckDataTypeFor3510::CheckGm2L1ScaleDataType<T, U>();
    }

    template <const CopyGM2L1Trait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto srcRowShape = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout);
        uint32_t srcColShape = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
        uint16_t srcBColStride =
            GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout);
        uint16_t dstBColStride =
            GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout);

        uint16_t dnNum = 1;
        uint16_t nValue = srcRowShape >> 1; // use b16 for DN2NZ, so nValue = srcRowShape / 2
        uint16_t dValue = srcColShape;
        uint16_t dstNzNStride = 1;

        uint64_t loop1SrcStride = srcBColStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;                             // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstBColStride * sizeof(type) / C0_SIZE<>; // loop3_dst_stride = dst_nz_c0_Stride
        uint16_t loop4DstStride = 0;
        uint8_t cacheMode = src.Engine().GetCacheMode();
        // fp8 scale use b16 for movement
        CopyGmToCbufMultiDn2nzInstr::CopyGmToCbufMultiDn2nz(
            (__cbuf__ half*)(dst.Data().Get()), (__gm__ half*)(src.Data().Get()), dnNum, loop2DstStride, loop3DstStride,
            loop4DstStride, loop1SrcStride, cacheMode, nValue, dValue, 0, false);
    }
};

} // namespace Te
} // namespace AscendC

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
