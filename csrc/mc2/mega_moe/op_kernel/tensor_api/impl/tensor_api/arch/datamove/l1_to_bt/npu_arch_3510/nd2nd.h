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
#warning "nd2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_BT_NPU_ARCH_3510_ND2ND_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_BT_NPU_ARCH_3510_ND2ND_H

#include "impl/tensor_api/arch/datamove/l1_to_bt/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

struct CopyL12BTTrait {};

class DataCopyL12BT3510 {
public:
    template <const CopyL12BTTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src) {
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const CopyL12BTTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckLayoutPattern<T, U>();
        CheckDataTypeFor3510::CheckL12BtDataType<T, U>();
    }

    template <const CopyL12BTTrait& trait, typename T, typename U>
    __aicore__ inline static auto DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t srcCol = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
        uint16_t srcRow = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(srcLayout);
        uint16_t dstCol = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(dstLayout);
        uint16_t dstRow = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout);

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        bool convControl = false;
        if (IsOneOfAttrV<srcType, half> && IsOneOfAttrV<dstType, float>) {
            convControl = true;
        }

        uint16_t blockLen = Std::ceil_division(srcCol, C0_ELEMENT<dstType>);
        if constexpr (IsOneOfAttrV<srcType, float, int32_t>) {
            blockLen = Std::ceil_align(blockLen, 2);
        }

        uint16_t blockCount = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout);
        uint16_t srcStride = (srcRow - srcCol) / C0_ELEMENT<srcType>;
        uint16_t dstStride = Std::ceil_align((dstRow - srcCol) / C0_ELEMENT<dstType>, 2);
        CopyL12BTInstr::DataCopy(dst, src, convControl, blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_BT_NPU_ARCH_3510_ND2ND_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
