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
    "impl/tensor_api/arch/cube/l1_to_fb/copy_impl/nd2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nd2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H

#include "impl/tensor_api/arch/cube/l1_to_fb/copy_impl/instruction.h"

namespace AscendC {
namespace Te {

class CopyL12FBND {
public:
    template <const CopyL12FBTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    {
        // Batch layouts carry a leading B axis: (B, (row, col)) -> depth 3,
        // (B, ((1, row), (1, col))) -> depth 5. Non-batch layouts are depth 2/4.
        constexpr auto srcDepth = NestingDepthV<decltype(src.Layout().Shape())>;
        if constexpr (srcDepth == THREE_DIM_DATA || srcDepth == FIVE_DIM_DATA) {
            BatchDataCopyImpl<trait, T, U>(dst, src);
        } else {
            DataCopyImpl<trait, T, U>(dst, src);
        }
    }

private:
    template <const CopyL12FBTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckDataType::CheckL12FbDataType<T, U>();
        CheckLayoutPattern<U, T>();
    }

    template <const CopyL12FBTrait& trait, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        constexpr uint32_t deqTensorAddrAlignValue = 128;
        constexpr uint16_t fbufBurstLenUnit = 64;

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t srcCol;
        uint16_t srcRow;
        uint16_t dstCol;
        uint16_t dstRow;
        uint16_t blockCount;
        if constexpr (IsSatisfiedPtnFormatV<U, NDLayoutPtn>) {
            srcCol = GetElement<AttrInfo::Shape, AttrInfo::Column>(srcLayout);
            srcRow = GetElement<AttrInfo::Stride, AttrInfo::Row>(srcLayout);
            blockCount = GetElement<AttrInfo::Shape, AttrInfo::Row>(srcLayout);
        } else {
            srcCol = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout);
            srcRow = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(srcLayout);
            blockCount = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout);
        }
        if constexpr (IsSatisfiedPtnFormatV<T, NDLayoutPtn>) {
            dstCol = GetElement<AttrInfo::Shape, AttrInfo::Column>(dstLayout);
            dstRow = GetElement<AttrInfo::Stride, AttrInfo::Row>(dstLayout);
        } else {
            dstCol = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(dstLayout);
            dstRow = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(dstLayout);
        }

        uint16_t blockLen = Std::ceil_align(srcCol * sizeof(srcType), deqTensorAddrAlignValue) / fbufBurstLenUnit;
        uint16_t srcStride = Std::ceil_division(srcRow * sizeof(srcType), C0_SIZE<>);
        uint16_t dstStride = Std::ceil_align(dstRow * sizeof(dstType), deqTensorAddrAlignValue) / fbufBurstLenUnit;

        CopyL12FBInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
    }

    // Batch case: layout is (B, (M, N)) with strides (sB, (sM, sN)). Per-matrix internal
    // compactness is assumed (sM == N), so a single matrix can be moved as one blockLen-sized block.
    // Four DataCopy params are derived directly from the batched layout:
    //   - GetElement<Shape, Row/Column> on the batched layout returns sub-matrix M/N (the
    //     SelectRowColTuples helper in is_format.h handles batch-axis stripping).
    //   - Get<0>(Shape/Stride) returns the batch size and per-matrix start-to-start stride.
    //   - When batches are also contiguous (srcBatchStride == M*N && dstBatchStride == M*N),
    //     fold the B blocks into a single B*M*N block; otherwise emit B blocks, one per matrix.
    //   - srcStride uses ceil_division(..., C0_SIZE), dstStride uses ceil_align(..., 128) / 64
    //     in keeping with the L1->FB instruction's stride unit conventions.
    template <const CopyL12FBTrait& trait, typename T, typename U>
    __aicore__ inline static void BatchDataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        constexpr uint32_t deqTensorAddrAlignValue = 128;
        constexpr uint16_t fbufBurstLenUnit = 64;

        auto srcLayout = src.Layout();
        auto dstLayout = dst.Layout();

        uint32_t srcBatchStride = Get<0>(srcLayout.Stride());
        uint32_t dstBatchStride = Get<0>(dstLayout.Stride());
        uint16_t batchSize = Get<0>(srcLayout.Shape());

        // Strip the leading B axis before calling GetElement, otherwise the dim=0/1 split below
        // would read the batch axis as row/column.
        auto inner = Te::Get<1>(srcLayout);

        uint32_t srcShapeRows;
        uint32_t srcShapeColumns;
        if constexpr (IsSatisfiedPtnFormatV<U, NDLayoutPtn>) {
            srcShapeRows = GetElement<AttrInfo::Shape, AttrInfo::Row>(inner);
            srcShapeColumns = GetElement<AttrInfo::Shape, AttrInfo::Column>(inner);
        } else {
            srcShapeRows = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(inner);
            srcShapeColumns = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(inner);
        }

        uint32_t matrixElems = srcShapeRows * srcShapeColumns;

        uint16_t blockCount;
        uint16_t blockLen;
        uint16_t srcStride;
        uint16_t dstStride;

        if (srcBatchStride == matrixElems && dstBatchStride == matrixElems) {
            // batches are contiguous on both sides: fold to one B*M*N block.
            blockCount = 1;
            blockLen = Std::ceil_align(batchSize * matrixElems * sizeof(srcType), deqTensorAddrAlignValue) /
                       fbufBurstLenUnit;
            srcStride = 0;
            dstStride = 0;
        } else {
            // batch-strided: B blocks, stride uses each side's native unit conversion.
            blockCount = batchSize;
            blockLen = Std::ceil_align(matrixElems * sizeof(srcType), deqTensorAddrAlignValue) / fbufBurstLenUnit;
            srcStride = Std::ceil_division(srcBatchStride * sizeof(srcType), C0_SIZE<>);
            dstStride =
                Std::ceil_align(dstBatchStride * sizeof(dstType), deqTensorAddrAlignValue) / fbufBurstLenUnit;
        }

        CopyL12FBInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_FB_COPY_IMPL_ND2ND_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
