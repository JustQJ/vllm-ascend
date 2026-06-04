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
    "impl/tensor_api/arch/datamove/gm_to_ub/npu_arch_3510/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_UB_NPU_ARCH_3510_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_UB_NPU_ARCH_3510_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/datamove/gm_to_ub/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

struct CopyGM2UBTrait {};

class DataCopyGM2UB3510 {
public:
    template <const CopyGM2UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    { Execute<trait>(dst, src); }

private:
    template <const CopyGM2UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Execute(const T& dst, const U& src)
    {
        using SRC_TYPE = typename U::elementType;
        using DST_TYPE = typename T::elementType;

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint8_t cacheMode = src.Engine().GetCacheMode();
        constexpr uint8_t leftPaddingCnt = 0;
        constexpr uint8_t rightPaddingCnt = 0;

        uint16_t blockCount = 0;
        uint32_t blockLen = 0;
        int64_t srcStride = 0;
        int64_t dstStride = 0;

        if constexpr (IsSatisfiedPtnFormatV<U, NDExtLayoutPtn> && IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) {
            blockCount = GetTotalRowShape(srcLayout);
            blockLen = GetTotalColumnShape(srcLayout) * sizeof(SRC_TYPE);

            srcStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(srcLayout) * sizeof(SRC_TYPE);
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout) * sizeof(DST_TYPE);

        } else if constexpr (IsSatisfiedPtnFormatV<U, DNExtLayoutPtn> && IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>) {
            blockCount = GetTotalColumnShape(srcLayout);
            blockLen = GetTotalRowShape(srcLayout) * sizeof(SRC_TYPE);

            srcStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout) * sizeof(SRC_TYPE);
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout) * sizeof(DST_TYPE);

        } else if constexpr (IsSatisfiedPtnFormatV<U, NZLayoutPtn> && IsSatisfiedPtnFormatV<T, NZLayoutPtn>) { // NZ format
            blockCount = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
            blockLen = GetTotalRowShape(srcLayout) * GetElement<AttrInfo::Shape, AttrInfo::Column, 0>(srcLayout) * sizeof(SRC_TYPE);

            srcStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(srcLayout) * sizeof(SRC_TYPE);
            dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout) * sizeof(DST_TYPE);
        } else {
            static_assert((IsSatisfiedPtnFormatV<U, NDExtLayoutPtn> && IsSatisfiedPtnFormatV<T, NDExtLayoutPtn>) || 
                          (IsSatisfiedPtnFormatV<U, DNExtLayoutPtn> && IsSatisfiedPtnFormatV<T, DNExtLayoutPtn>) || 
                          (IsSatisfiedPtnFormatV<U, NZLayoutPtn> && IsSatisfiedPtnFormatV<T, NZLayoutPtn>),
                          "Unsupported layout type combination for DataCopyGM2UB3510");
        }

        if constexpr (IsB4Type<SRC_TYPE>) {
            // move fp4 as b8, need to be divided by 2
            blockLen = blockLen >> 1;
            srcStride = srcStride >> 1;
        }

        if constexpr (IsB4Type<DST_TYPE>) {
            dstStride = dstStride >> 1;
        }

        CopyGmToUbufAlignV2Instr::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt, srcStride,
                                           dstStride, cacheMode);
        // ND和DN场景，需要保证UB上申请的空间和tensor的stride满足32字节对齐，否则CopyGmToUbufAlignV2会有问题，无法正确加载数据，导致数据错误
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_UB_NPU_ARCH_3510_DATA_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif