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
    "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/zn2nzb8b4_with_coord.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file zn2nzb8b4_with_coord.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_NPU_ARCH_3510_ZN2NZB8B4_WITH_COORD_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_NPU_ARCH_3510_ZN2NZB8B4_WITH_COORD_H

#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {
class LoadDataL12L0AZN2NZB8B4WithCoord3510 {

public:
    template <const CopyL12L0ATrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline static void Run(const T& dst, const U& src, const Coord& coord) {
        LoadDataImpl<TransTrait<trait, true>, T, U, Coord>(dst, src, coord);
    }

private:
    template <const CopyL12L0ATrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckLayoutPattern<T, U>();
        CheckDataTypeFor3510::CheckL12L0ADataType<T, U>();
    }

    template <const CopyL12L0ATrait& trait, typename T, typename U>
    __aicore__ inline static void LoadDataImplB4(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr int SHIFT_M_STEP_B4 = 2;
        constexpr int M_STEP_MIN_VAL_B4 = 4;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B4;
        mStep = M_STEP_MIN_VAL_B4;
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            auto sliceDst = dst(MakeCoord(MakeCoord(0, 0), MakeCoord(0, idx)));
            LoadCbufToCaS43510::LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            mStartPosition += M_STEP_MIN_VAL_B4;
        }
    }

    template <const CopyL12L0ATrait& trait, typename T, typename U>
    __aicore__ inline static void LoadDataImplB8(const T& dst, const U& src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        constexpr int SHIFT_M_STEP_B8 = 1;
        constexpr int M_STEP_MIN_VAL_B8 = 2;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            auto sliceDst = dst(MakeCoord(MakeCoord(0, 0), MakeCoord(0, idx)));
            LoadCbufToCa3510::LoadData<trait>(sliceDst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    }

    template <const CopyL12L0ATrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline static void LoadDataImpl(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint16_t mStartPosition = Std::get<1>(coord) / FRACTAL_FIXED;
        uint16_t kStartPosition = Std::get<0>(coord) / C0_ELEMENT<DstType>;
        auto m1 = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout) *
                  GetElement<AttrInfo::Shape, AttrInfo::Row, 0>(srcLayout) -
                  GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(dstLayout) *
                  GetElement<AttrInfo::Shape, AttrInfo::Row, 0>(dstLayout) - Std::get<0>(coord);
        auto mStep = GetElement<AttrInfo::Shape, AttrInfo::Column, 1>(srcLayout) *
                GetElement<AttrInfo::Shape, AttrInfo::Column, 0>(srcLayout) / FRACTAL_FIXED - mStartPosition;
        auto kStep = GetElement<AttrInfo::Shape, AttrInfo::Row, 1>(srcLayout) *
                GetElement<AttrInfo::Shape, AttrInfo::Row, 0>(srcLayout) / C0_ELEMENT<DstType> - kStartPosition;
        // Zn -> Nz
        constexpr uint32_t STRIDE_UNIT = C0_ELEMENT<DstType> * FRACTAL_FIXED;
        auto srcStride = GetElement<AttrInfo::Stride, AttrInfo::Row, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetElement<AttrInfo::Stride, AttrInfo::Column, 1>(dstLayout) / STRIDE_UNIT;
        if constexpr (IsB4Type<DstType>) {
            if (m1 < FRACTAL_FIXED) {
                LoadCbufToCaS43510::LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            } else {
                LoadDataImplB4<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        } else {
            if (m1 < FRACTAL_FIXED) {
                LoadCbufToCa3510::LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            } else {
                LoadDataImplB8<trait, T, U>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
            }
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_NPU_ARCH_3510_L12L0A_ZN2NZB8B4_WITH_COORD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif