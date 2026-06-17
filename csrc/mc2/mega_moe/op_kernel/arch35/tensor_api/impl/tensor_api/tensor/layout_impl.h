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
    "impl/tensor_api/tensor/layout_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/layout_method.h"
#include "impl/tensor_api/tensor/coord_index.h"
#include "impl/tensor_api/tensor/layout_fractal.h"
#include "impl/tensor_api/tensor/layout_frame.h"

namespace AscendC {
namespace Te {

struct MinOp {
    template <typename T, typename U>
    __aicore__ inline constexpr auto operator()(const T& src, const U& dst) const
    {
        return Std::min(src, dst);
    }
};

struct DiffOp {
    template <typename T, typename U>
    __aicore__ inline constexpr auto operator()(const T& shape, const U& coord) const
    {
        return shape - coord;
    }
};

template <typename Coord, typename LayoutType>
__aicore__ inline decltype(auto) MakeCoordLayout(const Coord& coord, const LayoutType& layout)
{
    using ShapeType = Std::remove_cvref_t<decltype(layout.Shape())>;
    using CoordType = Std::remove_cvref_t<Coord>;
    static_assert(IsLayoutV<LayoutType> && Std::is_tuple_v<CoordType>, "LayoutType must be Layout");
    static_assert(NestingDepthV<ShapeType> == NestingDepthV<CoordType> &&
        Std::tuple_size_v<ShapeType> == Std::tuple_size_v<CoordType>,
        "Shape and coord must have same tuple structure");
    auto coordShape = TransformTupleApply(layout.Shape(), coord, DiffOp{});
    using TraitType = GetLayoutTrait<LayoutType>;
    using PatternType = GetLayoutPattern<LayoutType>;
    return MakePatternLayout<PatternType, TraitType>(coordShape, layout.Stride());
}

template <typename Coord, typename LayoutType, typename SliceShape, Std::enable_if_t<!IsLayoutV<SliceShape>, int> = 0>
__aicore__ inline decltype(auto) MakeSliceLayout(const Coord& coord, const LayoutType& layout, const SliceShape& sliceShape)
{
    static_assert(IsLayoutV<LayoutType>, "LayoutType must be Layout");
    static_assert(Std::is_tuple_v<SliceShape>,"SliceShape must be a tuple");
    static_assert(NestingDepthV<SliceShape> == TWO_DIM_DATA, "Only Support Two Dim SliceShape");
    using OriginShape = Std::remove_cvref_t<decltype(layout.Shape())>;
    if constexpr (NestingDepthV<SliceShape> == NestingDepthV<OriginShape>
        && Std::tuple_size_v<SliceShape> == Std::tuple_size_v<OriginShape>) {
        auto srcRow = Std::get<0>(layout.Shape()) - Std::get<0>(coord);
        auto srcCol = Std::get<1>(layout.Shape()) - Std::get<1>(coord);
        auto realRow = Std::min(srcRow, Std::get<0>(sliceShape));
        auto realCol = Std::min(srcCol, Std::get<1>(sliceShape));
        using TraitType = GetLayoutTrait<LayoutType>;
        using PatternType = GetLayoutPattern<LayoutType>;
        return MakePatternLayout<PatternType, TraitType>(MakeShape(realRow, realCol), layout.Stride());
    } else {
        static_assert(NestingDepthV<OriginShape> == FOUR_DIM_DATA, "Only Support Four Dim Layout");
        auto innerRow = Std::get<0>(GetShape<0>(layout));
        auto innerCol = Std::get<0>(GetShape<1>(layout));

        auto srcRow = innerRow * Std::get<1>(GetShape<0>(layout)) - Std::get<0>(coord);
        auto srcCol = innerCol * Std::get<1>(GetShape<1>(layout)) - Std::get<1>(coord);

        auto realRow = Std::min(srcRow, Std::get<0>(sliceShape));
        auto realCol = Std::min(srcCol, Std::get<1>(sliceShape));
        using TraitType = GetLayoutTrait<LayoutType>;
        using PatternType = GetLayoutPattern<LayoutType>;
        return MakePatternLayout<PatternType, TraitType>(MakeFractalShape(MakeShape(realRow, realCol), MakeShape(innerRow, innerCol)), layout.Stride());
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
