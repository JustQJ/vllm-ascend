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
    "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scaleb_nn2nn.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file scaleb_nn2nn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_NPU_ARCH_3510_GM_TO_L1_SCALEB_NN2NN_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_NPU_ARCH_3510_GM_TO_L1_SCALEB_NN2NN_H

#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class CopyGmToCbufScaleBNn2Nn {
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

        auto srcShapeColB = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
        auto srcShapeColS = GetElement<AttrInfo::Shape, AttrInfo::Column, 0>(srcLayout);
        auto srcShapeRowB = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout);
        auto srcStrideColB = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout);
        auto srcStrideColS = GetElement<AttrInfo::Stride, AttrInfo::Column, 0>(srcLayout);

        auto dstStrideColB = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout);

        uint8_t cacheMode = src.Engine().GetCacheMode();

        // lprp mode, dst_stride % C0_SIZE should be 0
        // multi rows copy, dst non-contiguous case
        uint32_t blockCount = srcShapeColB;
        uint32_t blockLen = srcShapeRowB * sizeof(type) * srcShapeColS * srcStrideColS;
        uint64_t srcStride = srcStrideColB * sizeof(type);
        uint32_t dstStride = dstStrideColB * sizeof(type);

        uint8_t leftPaddingCnt = 0;
        uint8_t rightPaddingCnt = 0;
        CopyGmToCbufAlignV2Base::CopyGmToCbufAlignV2((__cbuf__ half*)(dst.Data().Get()),
                                                     (__gm__ half*)(src.Data().Get()), blockCount, blockLen,
                                                     leftPaddingCnt, rightPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
