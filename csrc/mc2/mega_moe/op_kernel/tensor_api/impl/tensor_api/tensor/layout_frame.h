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
    "impl/tensor_api/tensor/layout_frame.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_frame.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#include "impl/tensor_api/tensor/layout_pattern.h"

namespace AscendC {
namespace Te {

using LayoutFormatSet = TupleMap<
    Std::tuple<NZLayoutPtn, MakeNzFrameLayout>,
    Std::tuple<NDLayoutPtn, MakeNDFrameLayout>,
    Std::tuple<DNLayoutPtn, MakeDNFrameLayout>,
    Std::tuple<NDExtLayoutPtn, MakeNDExtFrameLayout>,
    Std::tuple<DNExtLayoutPtn, MakeDNExtFrameLayout>,
    Std::tuple<NNLayoutPtn, MakeNnFrameLayout>,
    Std::tuple<ZZLayoutPtn, MakeZzFrameLayout>,
    Std::tuple<ZNLayoutPtn, MakeZnFrameLayout>,
    Std::tuple<ScaleANDLayoutPtn, MakeScaleANDFrameLayout>,
    Std::tuple<ScaleADNLayoutPtn, MakeScaleADNFrameLayout>,
    Std::tuple<ScaleBNDLayoutPtn, MakeScaleBNDFrameLayout>,
    Std::tuple<ScaleBDNLayoutPtn, MakeScaleBDNFrameLayout>>;

template <typename T = uint16_t, size_t C0 = 32 / sizeof(T)>
struct LayoutTraitDefault {
    using type = T;
    static constexpr auto C0_ELEMENT = Std::Int<C0>{};
};

template <typename T = fp8_e8m0_t, size_t C0 = 2 / sizeof(T)>
struct LayoutTraitScale {
    using type = T;
    static constexpr auto C0_ELEMENT = Std::Int<C0>{};
};

template <typename T = fp4x2_e2m1_t, size_t C0 = 64 / sizeof(T)>
struct LayoutTraitFP4 {
    using type = T;
    static constexpr auto C0_ELEMENT = Std::Int<C0>{};
};

template <typename T, typename = void>
struct IsFrameLayoutTrait : Std::false_type {};

template <typename T>
struct IsFrameLayoutTrait<T, void_t<typename T::type, decltype(T::C0_ELEMENT)>> : Std::true_type {};

template <typename T>
constexpr bool IsFrameLayoutTraitV = IsFrameLayoutTrait<T>::value;

template <typename LayoutPattern, typename TraitType = LayoutTraitDefault<>,
    Std::enable_if_t<!IsIntegralConstantV<TraitType>, int> = 0, typename... Args>
__aicore__ inline decltype(auto) MakeFrameLayout(const Args&... args) {
    static_assert(IsFrameLayoutTraitV<TraitType>,
        "MakeFrameLayout<LayoutPattern, TraitType>(...) expects TraitType to define type and C0_ELEMENT.");
    using GetLayoutMakeFun = typename LayoutFormatSet::template Get<LayoutPattern>;
    static_assert(!Std::is_same_v<GetLayoutMakeFun, EmptyValue>, "Unsupported layout pattern.");
    return GetLayoutMakeFun::template Make<TraitType>(args...);
}

template <typename LayoutPattern, typename IntType,
    Std::enable_if_t<IsIntegralConstantV<IntType>, int> = 0, typename... Args>
__aicore__ inline decltype(auto) MakeFrameLayout(const Args&... args) {
    using GetLayoutMakeFun = typename LayoutFormatSet::template Get<LayoutPattern>;
    static_assert(!Std::is_same_v<GetLayoutMakeFun, EmptyValue>, "Unsupported layout pattern.");
    return GetLayoutMakeFun::template Make<LayoutTraitDefault<uint16_t, IntType::value>>(args...);
}

template <typename LayoutPattern, typename TraitType = LayoutTraitDefault<>>
struct FrameLayoutFormat {
    template <typename... Args>
    __aicore__ inline decltype(auto) operator()(const Args&... args) {
        return MakeFrameLayout<LayoutPattern, TraitType>(args...);
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
